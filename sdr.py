import os
import winsound
import time
import numpy as np
import librosa
import argparse
from scipy.signal import correlate
from mir_eval.separation import bss_eval_sources
import inquirer
from inquirer.themes import Default
from blessed import Terminal
from colorama import Fore, Style, just_fix_windows_console
from tabulate import tabulate
from concurrent.futures import ProcessPoolExecutor

'''
v0.1
'''
term = Terminal()
just_fix_windows_console()
os.system('title SDR计算小工具 v0.1    by 领航员未鸟')


class CustomTheme(Default):
    def __init__(self):
        super().__init__()
        self.Question.mark_color = term.yellow
        self.Question.brackets_color = term.normal
        self.Question.default_color = term.normal
        self.Editor.opening_prompt_color = term.bright_black
        self.Checkbox.selection_color = term.cyan
        self.Checkbox.selection_icon = ">"
        self.Checkbox.selected_icon = "[X]"
        self.Checkbox.selected_color = term.yellow + term.bold
        self.Checkbox.unselected_color = term.normal
        self.Checkbox.unselected_icon = "[ ]"
        self.Checkbox.locked_option_color = term.gray50
        self.List.selection_color = term.cyan
        self.List.selection_cursor = ">>"
        self.List.unselected_color = term.normal


def play_sound(sound_type):
    if sound_type == 1:
        frequencies = [1000, 1500, 1000, 2000]
        duration = 300
    elif sound_type == 2:
        frequencies = [1000, 1000]
        duration = 600
    else:
        frequencies = [1000, 1500, 1000, 2000]
        duration = 300

    for freq in frequencies:
        winsound.Beep(freq, duration)
        time.sleep(0.1)


def read_and_resample_audio(ref_path, est_paths):
    try:
        print(Fore.RED + '[INFO]' + Style.RESET_ALL + f' 读取并重采样所选音频...', end='', flush=True)
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

        print(Fore.GREEN + "Done!\n" + Style.RESET_ALL)

    except Exception as e:
        print(Fore.RED + "[ERROR]" + Style.RESET_ALL + f" 读取或重采样音频时出错: {e}")
        raise e

    return ref_audio, est_audios, target_sr


def align_audio(ref_audio, est_audio, sample_rate, est_path, window_size=None):
    """
    calculate cross-correlation for aligning the tracks while also matching their lengths.
    """
    est_name = os.path.basename(est_path)
    print_est_name = Fore.CYAN + est_name + Style.RESET_ALL
    print(Fore.RED + '[INFO]' + Style.RESET_ALL + f' 计算 {print_est_name} 时间偏移...', end='', flush=True)
    ref_channel = ref_audio[0] if ref_audio.ndim == 2 else ref_audio
    est_channel = est_audio[0] if est_audio.ndim == 2 else est_audio

    # Set segment length, defaulting to the first 60 seconds.
    if window_size is None:
        window_size = 60 * sample_rate

    if len(ref_channel) > window_size:
        ref_channel = ref_channel[:window_size]
    if len(est_channel) > window_size:
        est_channel = est_channel[:window_size]

    correlation = correlate(est_channel, ref_channel, mode='full')
    lag = np.argmax(correlation) - len(ref_channel) + 1

    print(Fore.GREEN + f"{lag}" + Style.RESET_ALL)
    print(Fore.RED + '[INFO]' + Style.RESET_ALL + f' 对齐音频...', end='', flush=True)

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
    print(Fore.GREEN + "Done!" + Style.RESET_ALL)

    return aligned_ref_audio, aligned_est_audio, est_name


def compute_sdr(ref_audio, est_audio, est_name):
    sdr, sir, sar, _ = bss_eval_sources(ref_audio, est_audio)
    return sdr, sir, sar, est_name


def process_audio(ref_path, est_paths, num_workers):
    results = []
    futures = []
    ref_audio, est_audios, max_sr = read_and_resample_audio(ref_path, est_paths)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i, est_audio in enumerate(est_audios):
            aligned_ref_audio, aligned_est_audio, est_name = align_audio(ref_audio, est_audio, max_sr, est_paths[i])
            print(Fore.RED + '[INFO]' + Style.RESET_ALL + f' 配对并送入进程池，SDR计算中...\n')
            future = executor.submit(compute_sdr, aligned_ref_audio, aligned_est_audio, est_name)
            futures.append(future)

    for future in futures:
        try:
            sdr, sir, sar, est_name = future.result()
            results.append((est_name, sdr, sir, sar))
        except Exception as exc:
            print(Fore.RED + "[ERROR]" + Style.RESET_ALL + f" 处理音频时出错: {exc}")

    return results


def scan_folder(folder_path):
    sample_files = [entry.name for entry in os.scandir(folder_path) if entry.is_file()]
    return sample_files


def select_ref_audio(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    choices = scan_folder(folder)

    questions = [
        inquirer.List('ref_audio',
                      message=Fore.GREEN + Style.DIM + f"请选择参考音频（存放于{folder}）" + Style.RESET_ALL,
                      choices=choices,
                      ),
    ]
    ref_name = inquirer.prompt(questions, theme=CustomTheme())

    ref_path = folder + '/' + ref_name['ref_audio']
    return ref_path


def select_est_audio(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)

    def ask_checkbox():
        choices = scan_folder(folder)
        questions = [
            inquirer.Checkbox('est_audio',
                              message=Fore.GREEN + Style.DIM + f"请选择估计音频（存放于{folder}，你可以从sample文件夹找到一些用于输入模型的样本）" + Style.RESET_ALL,
                              choices=choices,
                              ),
        ]
        est_name = inquirer.prompt(questions, theme=CustomTheme())

        if not est_name['est_audio']:
            print(f"{Fore.YELLOW}[WARN] {Style.RESET_ALL}你必须至少选择一个选项！")
            play_sound(2)
            return ask_checkbox()
        else:
            return est_name

    est_path = [f"{folder}/{filename}" for filename in ask_checkbox()['est_audio']]

    return est_path


def print_estimation_results(results):
    best_sdr_idx = np.argmax([np.mean(result[1]) for result in results])

    table_data = []
    for idx, (est_name, sdr, sir, sar) in enumerate(results):
        row = [
            est_name,
            f"{Fore.YELLOW}{sdr[0]:.2f} / {sdr[1]:.2f}{Style.RESET_ALL}",
            f"{Fore.CYAN}{sir[0]:.2f} / {sir[1]:.2f}{Style.RESET_ALL}",
            f"{Fore.MAGENTA}{sar[0]:.2f} / {sar[1]:.2f}{Style.RESET_ALL}"
        ]
        if idx == best_sdr_idx:
            row = [Fore.GREEN + Style.BRIGHT + str(item) + Style.RESET_ALL for item in row]
        table_data.append(row)

    headers = ["估计音频（Estimated Audio）", "SDR (L / R)", "SIR (L / R)", "SAR (L / R)"]
    table = tabulate(table_data, headers=headers, tablefmt="github")

    print(table)
    print(f"{Fore.RED}\n                          *请注意：该表格只反映针对此单一样本的评估，不代表模型综合性能。{Style.RESET_ALL}")


def print_bss_eval_descriptions():
    print(Style.BRIGHT + Fore.BLACK + "=" * 90)
    print("{:^90}".format("各参数含义"))
    print("=" * 90 + Style.RESET_ALL)

    print(f"{Fore.YELLOW}1. SDR (Source-to-Distortion Ratio) - 信号失真比{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.BLACK}   含义: 衡量源信号的总失真程度，SDR 值越大，分离后的信号越接近原始信号，表示分离质量越好。\n{Style.RESET_ALL}")

    print(f"{Fore.CYAN}2. SIR (Source-to-Interference Ratio) - 信号干扰比{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.BLACK}   含义: 衡量目标源与干扰源的分离效果，SIR 值越大，说明对干扰源的抑制越好，如分人声更不漏器乐。\n{Style.RESET_ALL}")

    print(f"{Fore.MAGENTA}3. SAR (Source-to-Artifacts Ratio) - 信号伪影比{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.BLACK}   含义: 衡量分离信号中由算法引入的伪影程度，SAR 值越大，说明算法引入的伪影越少，分离结果越好。")

    print("=" * 90 + Style.RESET_ALL + '\n')


def explain_bss_eval_sources_requirements():
    print(f"{'=' * 80}")
    print(f"{Style.BRIGHT}{Fore.BLUE} sdr-eval音频分离评估工具指南 - 如何准备和选择音频{Style.RESET_ALL}")
    print(f"{'=' * 80}\n")

    print(f"{Style.BRIGHT}{Fore.GREEN}1. 什么是参考音频 (reference_audio)?{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'-' * 80}{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.BLACK}   参考音频是从混音工程直接导出的目标声音，是我们期望的分离后理想的音频结果。{Style.RESET_ALL}\n")
    print(f"{Fore.YELLOW}· 人声分离模型：{Style.RESET_ALL}参考音频应该是 没有伴奏的人声。")
    print(f"{Fore.YELLOW}· 和声分离模型：{Style.RESET_ALL}参考音频应该是 没有伴奏的主唱人声，不包含和声。")
    print(f"{Fore.YELLOW}· 混响分离模型：{Style.RESET_ALL}参考音频应该是 没有混响的干净人声。")
    print(f"{Fore.YELLOW}· 降噪模型：{Style.RESET_ALL}    参考音频应该是 没有噪声的净的目标声音。")
    print(f"{Fore.CYAN}{'-' * 80}{Style.RESET_ALL}\n")

    print(f"{Style.BRIGHT}{Fore.GREEN}2. 什么是估计音频 (estimated_audio)?{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'-' * 80}{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.BLACK}   估计音频是分离模型输出的音频结果，用来计算和理想的参考音频相似程度以确定分离质量。{Style.RESET_ALL}\n")
    print(f"{Fore.YELLOW}· 人声分离模型：{Style.RESET_ALL}通常输入模型的是 整首歌，取其分离结果。")
    print(f"{Fore.YELLOW}· 和声分离模型：{Style.RESET_ALL}通常输入模型的是 包含和声的人声，取其分离结果。")
    print(f"{Fore.YELLOW}· 混响分离模型：{Style.RESET_ALL}通常输入模型的是 带有混响的人声，取其分离结果。")
    print(f"{Fore.YELLOW}· 降噪模型：{Style.RESET_ALL}    通常输入模型的是 带有噪音的目标声音，取其分离结果。")
    print(f"{Fore.CYAN}{'-' * 80}{Style.RESET_ALL}\n")

    print(f"{Style.BRIGHT}{Fore.GREEN}3. 其他重要提示{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'-' * 80}{Style.RESET_ALL}")
    print(f"{Fore.RED} 保证音频质量：{Style.RESET_ALL}参考音频和输入分离模型的音频 不可使用分离出的音频。")
    print(f"{Fore.RED} 保证音频对应：{Style.RESET_ALL}参考音频和输入分离模型的音频 应来自同一个混音工程的不同导出。")
    print(f"{Fore.CYAN}{'-' * 80}{Style.RESET_ALL}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_folder", type=str, default='reference_audio', help="folder with reference audio to process")
    parser.add_argument("--estimated_folder", type=str, default='estimated_audio', help="folder with estimated audio to process")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers used for SDR computation")
    args = parser.parse_args()

    explain_bss_eval_sources_requirements()
    reference_path = select_ref_audio(args.reference_folder)
    estimated_path = select_est_audio(args.estimated_folder)

    start_time = time.time()

    results = process_audio(reference_path, estimated_path, args.num_workers)

    print(f"{Fore.RED}[INFO]{Style.RESET_ALL} 计算完成！消耗时间："+Fore.GREEN+"{:.2f} sec".format(time.time() - start_time)+Style.RESET_ALL)
    print_bss_eval_descriptions()
    print_estimation_results(results)


if __name__ == '__main__':
    main()
    play_sound(1)

