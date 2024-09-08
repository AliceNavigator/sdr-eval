import os
import time
import numpy as np
import librosa
from scipy.signal import correlate
from mir_eval.separation import bss_eval_sources
import gradio as gr
from tabulate import tabulate
from concurrent.futures import ProcessPoolExecutor
# from colorama import just_fix_windows_console

# just_fix_windows_console()


def explain_bss_eval_sources_requirements():
    return """
    <h3>什么是参考音频 (reference_audio)?</h3>
    <p>参考音频是从混音工程直接导出的目标声音，是我们期望的分离后理想的音频结果。</p>
    <ul>
    <li><strong>人声分离模型</strong>：参考音频应该是没有伴奏的人声。</li>
    <li><strong>和声分离模型</strong>：参考音频应该是没有伴奏的主唱人声，不包含和声。</li>
    <li><strong>混响分离模型</strong>：参考音频应该是没有混响的干净人声。</li>
    <li><strong>降噪模型</strong>：参考音频应该是没有噪声的干净的目标声音。</li>
    </ul>

    <h3>什么是估计音频 (estimated_audio)?</h3>
    <p>估计音频是分离模型输出的音频结果，用来计算和理想的参考音频相似程度以确定分离质量。</p>
    <ul>
    <li><strong>人声分离模型</strong>：通常输入模型的是整首歌，取其分离结果。</li>
    <li><strong>和声分离模型</strong>：通常输入模型的是包含和声的人声，取其分离结果。</li>
    <li><strong>混响分离模型</strong>：通常输入模型的是带有混响的人声，取其分离结果。</li>
    <li><strong>降噪模型</strong>：通常输入模型的是带有噪声的目标声音。</li>
    </ul>

    <h3>其他重要提示</h3>
    <ul>
    <li><strong>保证音频质量</strong>：参考音频和输入分离模型的音频不可使用分离出的音频。</li>
    <li><strong>保证音频对应</strong>：参考音频和输入分离模型的音频应来自同一个混音工程的不同导出。</li>
    </ul>
    """


def print_bss_eval_descriptions():
    return """
    <h3>各参数含义</h3>
    <ol>
    <li><strong>SDR (Source-to-Distortion Ratio) - 信号失真比</strong>
       <p>含义: 衡量源信号的总失真程度，SDR 值越大，分离后的信号越接近原始信号，表示分离质量越好。</p></li>
    <li><strong>SIR (Source-to-Interference Ratio) - 信号干扰比</strong>
       <p>含义: 衡量目标源与干扰源的分离效果，SIR 值越大，说明对干扰源的抑制越好，如分离人声时器乐分离效果越好。</p></li>
    <li><strong>SAR (Source-to-Artifacts Ratio) - 信号伪影比</strong>
       <p>含义: 衡量分离信号中由算法引入的伪影程度，SAR 值越大，说明算法引入的伪影越少，分离结果越好。</p></li>
    </ol>
    """


def read_and_resample_audio(ref_path, est_paths):
    ref_audio, ref_sr = librosa.load(ref_path, sr=None, mono=False)
    est_audios = []
    est_srs = []
    for est_path in est_paths:
        est_audio, est_sr = librosa.load(est_path, sr=None, mono=False)
        est_audios.append(est_audio)
        est_srs.append(est_sr)

    target_sr = max([ref_sr] + est_srs)
    if ref_sr != target_sr:
        ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=target_sr)
    for i in range(len(est_audios)):
        if est_srs[i] != target_sr:
            est_audios[i] = librosa.resample(est_audios[i], orig_sr=est_srs[i], target_sr=target_sr)

    ref_audio = ref_audio if ref_audio.ndim == 2 else np.vstack([ref_audio] * 2)
    est_audios = [audio if audio.ndim == 2 else np.vstack([audio] * 2) for audio in est_audios]

    return ref_audio, est_audios, target_sr


def align_audio(ref_audio, est_audio, sample_rate, est_path, window_size=None):
    est_name = os.path.basename(est_path)
    ref_channel = ref_audio[0] if ref_audio.ndim == 2 else ref_audio
    est_channel = est_audio[0] if est_audio.ndim == 2 else est_audio
    if window_size is None:
        window_size = 60 * sample_rate

    if len(ref_channel) > window_size:
        ref_channel = ref_channel[:window_size]
    if len(est_channel) > window_size:
        est_channel = est_channel[:window_size]

    correlation = correlate(est_channel, ref_channel, mode='full')
    lag = np.argmax(correlation) - len(ref_channel) + 1

    if lag > 0:
        aligned_ref_audio = ref_audio[:, :len(ref_audio[0]) - lag]
        aligned_est_audio = est_audio[:, lag:]
    elif lag < 0:
        aligned_ref_audio = ref_audio[:, -lag:]
        aligned_est_audio = est_audio[:, :len(est_audio[0]) + lag]
    else:
        aligned_ref_audio = ref_audio
        aligned_est_audio = est_audio

    min_len = min(len(aligned_ref_audio[0]), len(aligned_est_audio[0]))
    aligned_ref_audio = aligned_ref_audio[:, :min_len]
    aligned_est_audio = aligned_est_audio[:, :min_len]

    return aligned_ref_audio, aligned_est_audio, est_name


def compute_sdr(ref_audio, est_audio, est_name):
    sdr, sir, sar, _ = bss_eval_sources(ref_audio, est_audio)
    return sdr, sir, sar, est_name


def process_audio(ref_path, est_paths, num_workers=4):
    results = []
    futures = []
    ref_audio, est_audios, max_sr = read_and_resample_audio(ref_path, est_paths)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i, est_audio in enumerate(est_audios):
            aligned_ref_audio, aligned_est_audio, est_name = align_audio(ref_audio, est_audio, max_sr, est_paths[i])
            future = executor.submit(compute_sdr, aligned_ref_audio, aligned_est_audio, est_name)
            futures.append(future)

    for future in futures:
        try:
            sdr, sir, sar, est_name = future.result()
            results.append((est_name, sdr, sir, sar))
        except Exception as exc:
            print(f"Error: {exc}")

    return results


def print_estimation_results(results):
    best_sdr_idx = np.argmax([np.mean(result[1]) for result in results])

    table_data = []
    for idx, (est_name, sdr, sir, sar) in enumerate(results):
        row = [
            est_name,
            f"{sdr[0]:.2f} / {sdr[1]:.2f}",
            f"{sir[0]:.2f} / {sir[1]:.2f}",
            f"{sar[0]:.2f} / {sar[1]:.2f}"
        ]
        table_data.append(row)

    headers = ["Estimated Audio", "SDR (L / R)", "SIR (L / R)", "SAR (L / R)"]
    table = tabulate(table_data, headers=headers, tablefmt="html")
    return table


def sdr_eval_ui(ref_audio_file, est_audio_files):
    ref_path = ref_audio_file.name
    est_paths = [f.name for f in est_audio_files]
    results = process_audio(ref_path, est_paths)
    result_table = print_estimation_results(results)
    return result_table


with gr.Blocks() as demo:
    gr.Markdown("# SDR计算小工具 v0.1")
    gr.Markdown("#### GitHub 地址: https://github.com/AliceNavigator/sdr-eval | by 领航员未鸟")

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML(explain_bss_eval_sources_requirements())

        with gr.Column(scale=1):
            gr.HTML("<hr>")  # 添加分隔线
            gr.HTML(print_bss_eval_descriptions())

    gr.Markdown("### 请上传音频文件进行SDR评估")
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### 上传参考音频 (Reference Audio):")
            ref_audio = gr.File(label="Upload Reference Audio", file_types=["audio"])

        with gr.Column():
            gr.Markdown("#### 上传估计音频 (Estimated Audio):")
            est_audio = gr.File(label="Upload Estimated Audio", file_count="multiple", file_types=["audio"])

    run_btn = gr.Button("Run SDR Evaluation")

    gr.Markdown("### 评估结果")
    result_display = gr.HTML()

    run_btn.click(fn=sdr_eval_ui, inputs=[ref_audio, est_audio], outputs=result_display)

if __name__ == "__main__":
    demo.launch()
