import dearpygui.dearpygui as dpg
import matplotlib
import numpy as np
import json
import os
import pywt
import time
import psutil
import dearpygui.dearpygui as dpg
import threading
import sounddevice as sd
import padasip as pa
import pathlib
import time
# import tensorflow as tf
import librosa
matplotlib.use('Agg')
from scipy.io.wavfile import read, write
from scipy.stats import pearsonr, entropy
from scipy.signal import butter, bessel, cheby1, filtfilt, freqz, lfilter, sosfilt, welch
from scipy.fft import fft, fftfreq
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from pathlib import Path
from torchmetrics.functional.audio import signal_noise_ratio
from typing import Union, Tuple

# from tensorflow.python.keras.layers import *
# from keras.models import Sequential, load_model 
# from keras.layers import Dense, Input, Flatten, Reshape
# from sklearn.preprocessing import MinMaxScaler


CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
CONFIG_FILE = "window_config.json"
COMBINED_PATH = os.path.join(CURRENT_DIR, CONFIG_FILE)
window_config = {}

audio_file_original = None  # Oryginalny, niezaszumiony sygnal
sampling_rate_original = None  # Oryginalny, niezaszumiony sygnal - sampling rate
audio_file_path_original = None 

audio_file = None  # Tutaj bedzie zapisany wczytany plik audio, ktory bedzie podlegal odszumianiu
sampling_rate = None
audio_file_path = None
audio_file_filtered_save_path = None

audio_file_filtered = None  # Tutaj bedzie zapisany nasz przefiltrowany plik audio
sampling_rate_filtered = None
audio_file_path_filtered = None

# GUI Setup
dpg.create_context()
dpg.create_viewport(title='Denoiser App', width=1600, height=900)
dpg.toggle_viewport_fullscreen()
dpg.maximize_viewport()

def log_error(s):   # nice log xD
    print(s)

def get_audio_file_path_filtered():
    global audio_file_path_filtered
    return audio_file_path_filtered

###############################################################################################################################################################

#  _______   ________  __    __   ______   __    __  __       __   ______   _______   __    __ 
# /       \ /        |/  \  /  | /      \ /  |  /  |/  \     /  | /      \ /       \ /  |  /  |
# $$$$$$$  |$$$$$$$$/ $$  \ $$ |/$$$$$$  |$$ |  $$ |$$  \   /$$ |/$$$$$$  |$$$$$$$  |$$ | /$$/ 
# $$ |__$$ |$$ |__    $$$  \$$ |$$ |  $$/ $$ |__$$ |$$$  \ /$$$ |$$ |__$$ |$$ |__$$ |$$ |/$$/  
# $$    $$< $$    |   $$$$  $$ |$$ |      $$    $$ |$$$$  /$$$$ |$$    $$ |$$    $$< $$  $$<   
# $$$$$$$  |$$$$$/    $$ $$ $$ |$$ |   __ $$$$$$$$ |$$ $$ $$/$$ |$$$$$$$$ |$$$$$$$  |$$$$$  \  
# $$ |__$$ |$$ |_____ $$ |$$$$ |$$ \__/  |$$ |  $$ |$$ |$$$/ $$ |$$ |  $$ |$$ |  $$ |$$ |$$  \ 
# $$    $$/ $$       |$$ | $$$ |$$    $$/ $$ |  $$ |$$ | $/  $$ |$$ |  $$ |$$ |  $$ |$$ | $$  |
# $$$$$$$/  $$$$$$$$/ $$/   $$/  $$$$$$/  $$/   $$/ $$/      $$/ $$/   $$/ $$/   $$/ $$/   $$/ 
                                                                                             
###############################################################################################################################################################

# Globalne zmienne do pomiaru
filter_start_time = None
start_cpu_percent = None
start_memory_info = None

# Pobieramy aktualny proces
process = psutil.Process()

usage_data = {"cpu": [], "memory": []}
stop_flag = None 
usage_data = None 
monitor_thread = None

# for monitoring CPU and RAM usage
def start_resource_monitor(interval=0.1):
    global usage_data 
    usage_data = {"cpu": [], "memory": []}
    stop_flag = threading.Event()

    def monitor():
        process = psutil.Process(os.getpid())
        while not stop_flag.is_set():
            usage_data["cpu"].append(psutil.cpu_percent(interval=None))
            usage_data["memory"].append(process.memory_info().rss / (1024 ** 2))  # MB
            time.sleep(interval)

    thread = threading.Thread(target=monitor)
    thread.start()
    return stop_flag, usage_data, thread

def stop_resource_monitor(stop_flag, thread):
    stop_flag.set()
    thread.join()

def start_benchmark():
    global filter_start_time, start_cpu_percent, start_memory_info, stop_flag, usage_data, monitor_thread 
    filter_start_time = time.time()
    start_cpu_percent = psutil.cpu_percent(interval=None)
    start_memory_info = process.memory_info().rss  # RAM w bajtach uzywany przez nasz proces
    stop_flag, usage_data, monitor_thread = start_resource_monitor(interval=0.1)

def stop_benchmark_and_show_results():
    global filter_start_time, start_cpu_percent, start_memory_info, audio_file_original, audio_file_filtered, sampling_rate, sampling_rate_filtered, sampling_rate_original, \
            stop_flag, usage_data, monitor_thread 

    if filter_start_time is None:
        print("Benchmark nie zostal rozpoczety!")
        return

    stop_resource_monitor(stop_flag, monitor_thread)

    # Oblicz średnie
    avg_cpu = sum(usage_data["cpu"]) / len(usage_data["cpu"])
    avg_memory = sum(usage_data["memory"]) / len(usage_data["memory"])

    # Czas trwania
    elapsed_time = time.time() - filter_start_time

    # Aktualne zuzycie CPU i RAM
    end_cpu_percent = psutil.cpu_percent(interval=None)
    end_memory_info = process.memory_info().rss

    # Liczymy zmiany
    cpu_usage_diff = end_cpu_percent - start_cpu_percent
    memory_usage_diff = (end_memory_info - start_memory_info) / (1024 * 1024)  # MB

    benchmark_window_tag="BenchmarkWindow"
    if dpg.does_item_exist(benchmark_window_tag):
        dpg.delete_item(benchmark_window_tag)

    thd_value = calculate_thd(audio_file_filtered, sampling_rate_filtered)

    se_before = None
    se_after = None
    sfm_before = None
    sfm_after = None

    if audio_file_original is not None:
        se_before = calculate_spectral_entropy(audio_file_original, sampling_rate_original)
        sfm_before = calculate_spectral_flatness(audio_file_original, sampling_rate_original)

    se_after = calculate_spectral_entropy(audio_file_filtered, sampling_rate_filtered)
    sfm_after = calculate_spectral_flatness(audio_file_filtered, sampling_rate)


    snr_before = None
    snr_after = None
    # snr_bf = None
    # snr_af = None
    snr_segmental_before = None
    snr_segmental_after = None
    snr_cb_before = None
    snr_cb_after = None
    statistics_before = None
    statistics_after = None

    if (audio_file_original is not None and 
        audio_file is not None and 
        audio_file_filtered is not None):
        min_len = min(len(audio_file_original), len(audio_file), len(audio_file_filtered))

        snr_before = calculate_snr(audio_file_original[:min_len], audio_file[:min_len])
        snr_after = calculate_snr(audio_file_original[:min_len], audio_file_filtered[:min_len])

        # snr_bf = calculate_snr2(audio_file_original[:min_len], audio_file[:min_len], sr=sampling_rate_filtered, mode='mean')
        # snr_af = calculate_snr2(audio_file_original[:min_len], audio_file_filtered[:min_len], sr=sampling_rate_filtered, mode='mean')

        snr_segmental_before = compute_segmental_snr(audio_file_original[:min_len], audio_file[:min_len], sampling_rate_filtered)
        snr_segmental_after = compute_segmental_snr(audio_file_original[:min_len], audio_file_filtered[:min_len], sampling_rate_filtered)
        
        snr_cb_before = compute_critical_band_snr(audio_file_original[:min_len], audio_file[:min_len], sampling_rate_filtered)
        snr_cb_after = compute_critical_band_snr(audio_file_original[:min_len], audio_file_filtered[:min_len], sampling_rate_filtered)

        statistics_before = calculate_signal_metrics(audio_file_original[:min_len], audio_file[:min_len])
        statistics_after = calculate_signal_metrics(audio_file_original[:min_len], audio_file_filtered[:min_len])
    else:
        if audio_file is None or audio_file_filtered is None:
            print("Brak danych: zaladuj plik wejsciowy i przefiltruj go.")
        elif audio_file_original is None:
            print("Nie wczytano oryginalnego (niezaszumionego) sygnalu, pomijanie liczenia SNR!")

    # Tworzymy popup
    with dpg.window(tag=benchmark_window_tag, label="Statystyki filtracji"):
        apply_window_geometry(benchmark_window_tag, default_pos=(820, 10), default_size=(500, 250))

        value_color = [0, 255, 0]  # Kolor wartości

        with dpg.group(horizontal=True):
            # Kolumna 1: Zasoby
            with dpg.group():
                dpg.add_text("Zasoby sprzetowe, THD, Spectral Entropy", color=[200, 200, 255])
                with dpg.group(horizontal=True):
                    with dpg.group():
                        dpg.add_text("Czas filtracji:")
                        dpg.add_text("Srednie zuzycie CPU:")
                        dpg.add_text("Srednie zuzycie RAM:")
                        dpg.add_text("THD:")
                        if audio_file_original is not None:
                            dpg.add_text("Spectral Entropy (Original):")
                        dpg.add_text("Spectral Entropy (Filtered):")

                        if audio_file_original is not None:
                            dpg.add_text("Spectral Flatness Measure (Original):")
                        dpg.add_text("Spectral Flatness Measure (Filtered):")

                    with dpg.group():
                        dpg.add_text(f"{elapsed_time:.3f} s", color=value_color)
                        dpg.add_text(f"{avg_cpu:.2f}%", color=value_color)
                        dpg.add_text(f"{avg_memory:+.2f} MB", color=value_color)
                        dpg.add_text(f"{thd_value:.2f}%", color=value_color)
                        if audio_file_original is not None:
                            dpg.add_text(f"{se_before:.4f}", color=value_color)
                        dpg.add_text(f"{se_after:.4f}", color=value_color)

                        if audio_file_original is not None:
                            dpg.add_text(f"{sfm_before:.4f}", color=value_color)
                        dpg.add_text(f"{sfm_after:.4f}", color=value_color)

            if (audio_file_original is not None and 
                audio_file is not None and 
                audio_file_filtered is not None):

                # Kolumna 2: SNR    
                with dpg.group():
                    dpg.add_text("SNR", color=[200, 255, 200])
                    with dpg.group(horizontal=True):
                        with dpg.group():
                            dpg.add_text("SNR przed filtracja (before):")
                            dpg.add_text("SNR po filtracji (after):")
                            # dpg.add_text("Sredni SNR (before):")
                            # dpg.add_text("Sredni SNR (after):")
                            dpg.add_text("SNR segmentowy (before):")
                            dpg.add_text("SNR segmentowy (after):")
                            dpg.add_text("SNR pasm krytycznych (before):")
                            dpg.add_text("SNR pasm krytycznych (after):")
                            dpg.add_text("PSNR (before):")
                            dpg.add_text("PSNR (after):")
                        with dpg.group():
                            dpg.add_text(f"{snr_before:.2f} dB", color=value_color)
                            dpg.add_text(f"{snr_after:.2f} dB", color=value_color)
                            # dpg.add_text(f"{snr_bf:.2f} dB", color=value_color)
                            # dpg.add_text(f"{snr_af:.2f} dB", color=value_color)
                            dpg.add_text(f"{snr_segmental_before:.2f} dB", color=value_color)
                            dpg.add_text(f"{snr_segmental_after:.2f} dB", color=value_color)
                            dpg.add_text(f"{snr_cb_before:.2f} dB", color=value_color)
                            dpg.add_text(f"{snr_cb_after:.2f} dB", color=value_color)
                            dpg.add_text(f"{statistics_before['PSNR']:.2f} dB", color=value_color)
                            dpg.add_text(f"{statistics_after['PSNR']:.2f} dB", color=value_color)

                # Kolumna 3: MSE, NRMSE, korelacja
                with dpg.group():
                    dpg.add_text("Bledy i korelacja", color=[255, 200, 200])
                    with dpg.group(horizontal=True):
                        with dpg.group():
                            dpg.add_text("MSE (before):")
                            dpg.add_text("MSE (after):")
                            dpg.add_text("NRMSE (before):")
                            dpg.add_text("NRMSE (after):")
                            dpg.add_text("Correlation (before):")
                            dpg.add_text("Correlation (after):")
                        with dpg.group():
                            dpg.add_text(f"{statistics_before['MSE']:.4f}", color=value_color)
                            dpg.add_text(f"{statistics_after['MSE']:.4f}", color=value_color)
                            dpg.add_text(f"{statistics_before['NRMSE']:.4f}", color=value_color)
                            dpg.add_text(f"{statistics_after['NRMSE']:.4f}", color=value_color)
                            dpg.add_text(f"{statistics_before['Correlation Coefficient']:.2f}", color=value_color)
                            dpg.add_text(f"{statistics_after['Correlation Coefficient']:.2f}", color=value_color)

    # Resetuj zmienne
    filter_start_time = None
    start_cpu_percent = None
    start_memory_info = None

# THD mierzy znieksztalcenia sygnalu, ktore powstaja jako harmoniczne — im nizszy THD, tym mniej znieksztalcony sygnal.
def calculate_thd(signal, fs, fundamental_freq=None):
    # FFT
    N = len(signal)
    yf = fft(signal)
    yf = np.abs(yf[:N // 2])  # tylko dodatnie czestotliwosci
    freqs = fftfreq(N, 1 / fs)[:N // 2]

    # Znajdz skladowa podstawowa
    if fundamental_freq is None:
        idx_f1 = np.argmax(yf)
    else:
        idx_f1 = np.argmin(np.abs(freqs - fundamental_freq))

    fundamental = yf[idx_f1]
    
    # Harmoniczne to wielokrotnosci skladowej podstawowej
    harmonics = []
    for i in range(2, 6):  # 2. do 5. harmonicznej
        target_freq = i * freqs[idx_f1]
        idx = np.argmin(np.abs(freqs - target_freq))
        harmonics.append(yf[idx])

    harmonic_power = np.sum(np.square(harmonics))
    thd = np.sqrt(harmonic_power) / fundamental

    return thd * 100  # jako %

def calculate_spectral_entropy(signal, sampling_rate, base=2):
    """
    Oblicza Spectral Entropy sygnału.

    Parameters:
        signal (np.ndarray): sygnał 1D.
        sampling_rate (int): częstotliwość próbkowania.
        method (str): 'fft' lub 'welch'.
        nperseg (int): segment size dla metody Welch'a.
        base (float): podstawa logarytmu (domyślnie 2 dla bitów).

    Returns:
        float: wartość entropii widmowej.
    """
    # Widmo mocy z FFT
    fft_vals = np.fft.fft(signal)
    psd = np.abs(fft_vals) ** 2
   
    # Normalizacja do rozkładu prawdopodobieństwa
    psd = psd[:len(psd)//2]  # tylko dodatnie czestotliwosci
    psd_sum = np.sum(psd)
    if psd_sum == 0:
        return 0.0
    psd_norm = psd / psd_sum

    # Obliczenie entropii
    spectral_entropy = entropy(psd_norm, base=base)
    return spectral_entropy

def calculate_spectral_flatness(signal, sampling_rate, nperseg=256):
    """
    Oblicza Spectral Flatness Measure (SFM) sygnału.

    Parameters:
        signal (np.ndarray): Sygnał wejściowy.
        sampling_rate (int): Częstotliwość próbkowania.
        nperseg (int): Długość segmentu dla metody Welch'a.

    Returns:
        float: Wartość Spectral Flatness.
    """
    freqs, psd = welch(signal, fs=sampling_rate, nperseg=nperseg)

    psd = psd + 1e-12  # zapobiegamy log(0) i dzieleniu przez 0
    geometric_mean = np.exp(np.mean(np.log(psd)))
    arithmetic_mean = np.mean(psd)

    spectral_flatness = geometric_mean / arithmetic_mean
    return spectral_flatness

def calculate_MSE(reference_signal, test_signal):
    reference_signal = np.asarray(reference_signal, dtype=np.float64)
    test_signal = np.asarray(test_signal, dtype=np.float64)

    if len(reference_signal) != len(test_signal):
        raise ValueError("Sygnały musza mieć tę sama długość!")

    mse = mean_squared_error(reference_signal, test_signal)
    print(f"MSE: {mse}")

    return mse

def calculate_RMSE(reference_signal, test_signal):
    reference_signal = np.asarray(reference_signal, dtype=np.float64)
    test_signal = np.asarray(test_signal, dtype=np.float64)

    if len(reference_signal) != len(test_signal):
        raise ValueError("Sygnały musza mieć tę sama długość!")

    rmse = root_mean_squared_error(reference_signal, test_signal)
    print(f"RMSE: {rmse}")

    return rmse

def calculate_NRMSE(reference_signal, test_signal):
    reference_signal = np.asarray(reference_signal, dtype=np.float64)
    test_signal = np.asarray(test_signal, dtype=np.float64)

    if len(reference_signal) != len(test_signal):
        raise ValueError("Sygnały musza mieć tę sama długość!")

    nrmse = root_mean_squared_error(reference_signal, test_signal) / (np.max(reference_signal) - np.min(reference_signal) + 1e-12)
    print(f"NRMSE: {nrmse}")

    return nrmse

def calculate_PSNR(reference_signal, test_signal):
    reference_signal = np.asarray(reference_signal, dtype=np.float64)
    test_signal = np.asarray(test_signal, dtype=np.float64)

    if len(reference_signal) != len(test_signal):
        raise ValueError("Sygnały musza mieć tę sama długość!")

    mse = calculate_MSE(reference_signal, test_signal)

    # PSNR (Peak Signal-to-Noise Ratio)
    peak = np.max(np.abs(reference_signal))
    psnr = 20 * np.log10(peak / (np.sqrt(mse) + 1e-12))  # dB

    return psnr

def calculate_correlation(reference_signal, test_signal):
    reference_signal = np.asarray(reference_signal, dtype=np.float64)
    test_signal = np.asarray(test_signal, dtype=np.float64)

    if len(reference_signal) != len(test_signal):
        raise ValueError("Sygnały musza mieć tę sama długość!")

    # Współczynnik korelacji
    corr, _ = pearsonr(reference_signal, test_signal)

    return corr

def calculate_signal_metrics(reference_signal, test_signal):
    """
    Porównuje dwa sygnały: oryginalny (reference) i przefiltrowany/testowy (test).
    Zwraca MSE, NRMSE, PSNR i współczynnik korelacji.
    """
    reference_signal = np.asarray(reference_signal, dtype=np.float64)
    test_signal = np.asarray(test_signal, dtype=np.float64)

    if len(reference_signal) != len(test_signal):
        raise ValueError("Sygnały musza mieć tę sama długość!")

    mse = np.mean((reference_signal - test_signal) ** 2)
    rmse = np.sqrt(mse)
    nrmse = rmse / (np.max(reference_signal) - np.min(reference_signal) + 1e-12)

    # PSNR (Peak Signal-to-Noise Ratio)
    peak = np.max(np.abs(reference_signal))
    psnr = 20 * np.log10(peak / (np.sqrt(mse) + 1e-12))  # dB values

    # Współczynnik korelacji
    corr, _ = pearsonr(reference_signal, test_signal)

    return {
        "MSE": mse,
        "NRMSE": nrmse,
        "PSNR": psnr,
        "Correlation Coefficient": corr
    }

###############################################################################################################################################################

#  ______  __    __  _______   ______   ______    ______   ________  ______   _______  
# /      |/  \  /  |/       \ /      | /      \  /      \ /        |/      \ /       \ 
# $$$$$$/ $$  \ $$ |$$$$$$$  |$$$$$$/ /$$$$$$  |/$$$$$$  |$$$$$$$$//$$$$$$  |$$$$$$$  |
#   $$ |  $$$  \$$ |$$ |  $$ |  $$ |  $$ |  $$/ $$ |__$$ |   $$ |  $$ |  $$ |$$ |__$$ |
#   $$ |  $$$$  $$ |$$ |  $$ |  $$ |  $$ |      $$    $$ |   $$ |  $$ |  $$ |$$    $$< 
#   $$ |  $$ $$ $$ |$$ |  $$ |  $$ |  $$ |   __ $$$$$$$$ |   $$ |  $$ |  $$ |$$$$$$$  |
#  _$$ |_ $$ |$$$$ |$$ |__$$ | _$$ |_ $$ \__/  |$$ |  $$ |   $$ |  $$ \__$$ |$$ |  $$ |
# / $$   |$$ | $$$ |$$    $$/ / $$   |$$    $$/ $$ |  $$ |   $$ |  $$    $$/ $$ |  $$ |
# $$$$$$/ $$/   $$/ $$$$$$$/  $$$$$$/  $$$$$$/  $$/   $$/    $$/    $$$$$$/  $$/   $$/ 
                                                                                     
###############################################################################################################################################################
# Globalne zmienne
processing_indicator = None
processing_flag = False

def start_processing_indicator():
    global processing_indicator, processing_flag
    processing_flag = True

    if dpg.does_item_exist("ProcessingDrawList"):
        dpg.delete_item("ProcessingDrawList")

    # Dodajemy napis bezposrednio do viewport
    with dpg.viewport_drawlist(tag="ProcessingDrawList"):
        processing_indicator = dpg.draw_text((10, dpg.get_viewport_client_height() - 30), "🔄 Trwa filtracja...", size=20, color=(255, 0, 0))

    def blink():
        visible = True
        while processing_flag:
            dpg.configure_item(processing_indicator, show=visible)
            visible = not visible
            time.sleep(0.5)
        
        if dpg.does_item_exist("ProcessingDrawList"):
            dpg.delete_item("ProcessingDrawList")

    threading.Thread(target=blink, daemon=True).start()

def stop_processing_indicator():
    global processing_flag
    processing_flag = False
    if dpg.does_item_exist("ProcessingDrawList"):
        dpg.delete_item("ProcessingDrawList")

    
###############################################################################################################################################################

#   ______   __    __  __    __   ______  
#  /      \ /  |  /  |/  \  /  | /      \ 
# /$$$$$$  |$$ | /$$/ $$  \ $$ |/$$$$$$  |
# $$ |  $$ |$$ |/$$/  $$$  \$$ |$$ |__$$ |
# $$ |  $$ |$$  $$<   $$$$  $$ |$$    $$ |
# $$ |  $$ |$$$$$  \  $$ $$ $$ |$$$$$$$$ |
# $$ \__$$ |$$ |$$  \ $$ |$$$$ |$$ |  $$ |
# $$    $$/ $$ | $$  |$$ | $$$ |$$ |  $$ |
#  $$$$$$/  $$/   $$/ $$/   $$/ $$/   $$/ 
                                        
###############################################################################################################################################################

def load_window_config():
    global window_config
    if os.path.exists(COMBINED_PATH):
        with open(COMBINED_PATH, "r") as f:
            try:
                window_config = json.load(f)
            except json.JSONDecodeError:
                window_config = {}

def save_window_config():
    print("[save_window_config] Saving config to file...")
    print(window_config)
    try:
        with open(COMBINED_PATH, "w") as f:
            json.dump(window_config, f, indent=4)
        print(f"[save_window_config] Config saved to {COMBINED_PATH}")
    except Exception as e:
        print(f"[save_window_config] Error saving config: {e}")

def apply_window_geometry(tag, default_pos=(100, 100), default_size=(400, 300)):
    config = window_config.get(tag, {})
    pos = config.get("pos", default_pos)
    size = config.get("size", default_size)

    dpg.set_item_pos(tag, pos)
    dpg.set_item_width(tag, size[0])
    dpg.set_item_height(tag, size[1])
    vieport_resized()

def store_geometry_on_close(sender, app_data, user_data):
    tag = user_data
    pos = dpg.get_item_pos(tag)
    width = dpg.get_item_width(tag)
    height = dpg.get_item_height(tag)
    window_config[tag] = {
        "pos": pos,
        "size": [width, height]
    }

    if sender != "save_layout":
        dpg.delete_item(tag)

def on_resize_or_move(sender, app_data, user_data):
    tag = user_data
    pos = dpg.get_item_pos(tag)
    width = dpg.get_item_width(tag)
    height = dpg.get_item_height(tag)
    window_config[tag] = {
        "pos": pos,
        "size": [width, height]
    }
    save_window_config()

###############################################################################################################################################################

#  __         ______   _______    ______   __       __   ______   __    __  ______  ________        _______   __        ______  __    __  __    __ 
# /  |       /      \ /       \  /      \ /  |  _  /  | /      \ /  \  /  |/      |/        |      /       \ /  |      /      |/  |  /  |/  |  /  |
# $$ |      /$$$$$$  |$$$$$$$  |/$$$$$$  |$$ | / \ $$ |/$$$$$$  |$$  \ $$ |$$$$$$/ $$$$$$$$/       $$$$$$$  |$$ |      $$$$$$/ $$ | /$$/ $$ |  $$ |
# $$ |      $$ |__$$ |$$ |  $$ |$$ |  $$ |$$ |/$  \$$ |$$ |__$$ |$$$  \$$ |  $$ |  $$ |__          $$ |__$$ |$$ |        $$ |  $$ |/$$/  $$ |  $$ |
# $$ |      $$    $$ |$$ |  $$ |$$ |  $$ |$$ /$$$  $$ |$$    $$ |$$$$  $$ |  $$ |  $$    |         $$    $$/ $$ |        $$ |  $$  $$<   $$ |  $$ |
# $$ |      $$$$$$$$ |$$ |  $$ |$$ |  $$ |$$ $$/$$ $$ |$$$$$$$$ |$$ $$ $$ |  $$ |  $$$$$/          $$$$$$$/  $$ |        $$ |  $$$$$  \  $$ |  $$ |
# $$ |_____ $$ |  $$ |$$ |__$$ |$$ \__$$ |$$$$/  $$$$ |$$ |  $$ |$$ |$$$$ | _$$ |_ $$ |_____       $$ |      $$ |_____  _$$ |_ $$ |$$  \ $$ \__$$ |
# $$       |$$ |  $$ |$$    $$/ $$    $$/ $$$/    $$$ |$$ |  $$ |$$ | $$$ |/ $$   |$$       |      $$ |      $$       |/ $$   |$$ | $$  |$$    $$/ 
# $$$$$$$$/ $$/   $$/ $$$$$$$/   $$$$$$/  $$/      $$/ $$/   $$/ $$/   $$/ $$$$$$/ $$$$$$$$/       $$/       $$$$$$$$/ $$$$$$/ $$/   $$/  $$$$$$/  
                                                                                                                                                 
###############################################################################################################################################################

# Wczytywanie pliku audio i pokazanie wykresow
def load_wav_file_callback(sender, app_data):
    global audio_file, audio_file_path, sampling_rate

    audio_file_path = app_data['file_path_name']
    sampling_rate, audio_data = read(audio_file_path)

    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]  # wybierz tylko lewy kanal (bez usredniania)

    # Konwersja do floatow - potrzebune do pozniej filtracji itd.
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0

    print("Loaded audio values:")
    print(audio_data)

    audio_file = audio_data
    audio_file_filtered_save_path = os.path.dirname(audio_file_path)
    print(f"Zaladowano plik: {audio_file_path}, dane: {audio_file.shape}")

    prepare_environment_for_filtering()

    call_create_audio_play_callback()
    show_plot_callback()
    show_spectrum_callback()

def load_original_wav_file_callback(sender, app_data):
    global audio_file_original, sampling_rate_original, audio_file_path_original

    audio_file_path_original = app_data['file_path_name']
    sampling_rate_original, data = read(audio_file_path_original)

    if len(data.shape) > 1:
        data = data[:, 0]  # wybierz tylko lewy kanal (bez usredniania)

    # Konwersja do floatow - potrzebune do pozniej filtracji itd.
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0

    print("Original loaded audio values:")
    print(data)
    print(sampling_rate_original)

    audio_file_original = data
    print(f"Zaladowano plik: {audio_file_path_original}, dane: {audio_file_original.shape}")


###############################################################################################################################################################

#  __       __  __      __  __    __  _______   ________   ______   __      __  
# /  |  _  /  |/  \    /  |/  |  /  |/       \ /        | /      \ /  \    /  |    
# $$ | / \ $$ |$$  \  /$$/ $$ | /$$/ $$$$$$$  |$$$$$$$$/ /$$$$$$  |$$  \  /$$/     
# $$ |/$  \$$ | $$  \/$$/  $$ |/$$/  $$ |__$$ |$$ |__    $$ \__$$/  $$  \/$$/      
# $$ /$$$  $$ |  $$  $$/   $$  $$<   $$    $$< $$    |   $$      \   $$  $$/       
# $$ $$/$$ $$ |   $$$$/    $$$$$  \  $$$$$$$  |$$$$$/     $$$$$$  |   $$$$/        
# $$$$/  $$$$ |    $$ |    $$ |$$  \ $$ |  $$ |$$ |_____ /  \__$$ |    $$ |        
# $$$/    $$$ |    $$ |    $$ | $$  |$$ |  $$ |$$       |$$    $$/     $$ |        
# $$/      $$/     $$/     $$/   $$/ $$/   $$/ $$$$$$$$/  $$$$$$/      $$/         
                                                                                             
###############################################################################################################################################################

AmplitudePlotHeight = -1
AmplitudePlotWidth = -1
SpectrumPlotHeight = -1
SpectrumPlotWidth = -1

def show_plot_callback():
    global audio_file, sampling_rate, audio_file_path
    if audio_file is None:
        print("Najpierw wczytaj plik!")
        return

    if not isinstance(audio_file, np.ndarray):
        print("Blad: dane audio nie sa numpy array!")
        return

    # Sprawdzenie stereo/mono
    if audio_file.ndim == 2:
        data_to_plot = audio_file[:, 0]
    else:
        data_to_plot = audio_file

    if data_to_plot.size == 0:
        print("Brak danych audio do wyswietlenia.")
        return

    # --- DOWN SAMPLING ---
    max_points = 5000
    if len(data_to_plot) <= max_points:
        downsampled_data = data_to_plot
        downsampled_x = np.arange(len(data_to_plot))
    else:
        factor = len(data_to_plot) // max_points
        downsampled_data = data_to_plot[::factor]
        downsampled_x = np.arange(0, len(data_to_plot), factor)

        # Czasami X moze byc dluzszy niz Y przez zaokraglenie — przycinamy
        min_len = min(len(downsampled_data), len(downsampled_x))
        downsampled_data = downsampled_data[:min_len]
        downsampled_x = downsampled_x[:min_len]

    window_tag="AmplitudeWindow"
    # --- USUWANIE STAREGO OKNA ---
    if dpg.does_item_exist(window_tag):
        dpg.delete_item(window_tag)

    # --- TWORZENIE NOWEGO OKNA Z WYKRESEM ---
    with dpg.window(label="Wykres Amplitudy", tag=window_tag, 
                    on_close=lambda: delete_item(window_tag)):
        apply_window_geometry(window_tag, default_pos=(820, 10), default_size=(500, 250))

        with dpg.plot(tag=f"{window_tag}Plot", label="Amplituda w czasie", height=AmplitudePlotHeight, width=AmplitudePlotWidth):
            dpg.add_plot_axis(dpg.mvXAxis, label="Probki")
            with dpg.plot_axis(dpg.mvYAxis, label="Amplituda") as y_axis:
                line_series_tag = "Amp2"
                if dpg.does_item_exist(line_series_tag):
                    dpg.set_value(line_series_tag, [downsampled_x.tolist(), downsampled_data.tolist()])
                else:
                    dpg.add_line_series(downsampled_x.tolist(), downsampled_data.tolist(), label="Amplituda", parent=y_axis)

        # Add popup context menu to the plot
        with dpg.popup(parent=f"{window_tag}Plot", mousebutton=dpg.mvMouseButton_Right):
            dpg.add_text("Plot Options")
            plot_file_path =  os.path.join(os.path.dirname(audio_file_path), f"{Path(audio_file_path).stem}_AMPLITUDE.png")
            dpg.add_button(label="Save Plot", callback=lambda: save_plot_fullscreen(f"{window_tag}Plot", plot_file_path))

def show_spectrum_callback():
    global audio_file, sampling_rate, audio_file_path
    if audio_file is None:
        print("Najpierw wczytaj plik!")
        return

    # Jesli dane sa typu int16, przeskaluj do floatow -1..1 tylko na potrzeby wyswietlenia
    if audio_file.dtype == np.int16:
        data_to_plot = audio_file.astype(np.float32) / 32767.0
    else:
        data_to_plot = np.copy(audio_file)

    # Obsluga mono/stereo
    if data_to_plot.ndim == 2:
        data_to_plot = data_to_plot[:, 0]  # Wybieramy pierwszy kanal

    if data_to_plot.size == 0:
        print("Brak danych do wyswietlenia.")
        return

    # --- FFT ---
    spectrum = np.abs(np.fft.fft(data_to_plot))
    freqs = np.fft.fftfreq(len(data_to_plot), d=1.0 / sampling_rate)

    # Uzywamy tylko dodatnich czestotliwosci
    half = len(freqs) // 2
    freqs = freqs[:half]
    spectrum = spectrum[:half]

    # --- DOWNSAMPLING --- (dla lepszej wydajnosci przy duzych plikach)
    max_points = 5000
    if len(freqs) > max_points:
        factor = len(freqs) // max_points
        spectrum = spectrum[::factor]
        freqs = freqs[::factor]

    # Upewniamy sie, ze X i Y maja te sama dlugosc
    min_len = min(len(freqs), len(spectrum))
    freqs = freqs[:min_len]
    spectrum = spectrum[:min_len]

    window_tag="SpectrumWindow" 
    if dpg.does_item_exist(window_tag):
        dpg.delete_item(window_tag)

    # --- TWORZENIE NOWEGO OKNA Z WYKRESEM ---
    with dpg.window(label="Widmo czestotliwosci (FFT) przed filtracja", tag=window_tag, 
                    on_close=lambda: delete_item(window_tag)):
        apply_window_geometry(window_tag, default_pos=(820, 270), default_size=(500, 250))

        with dpg.plot(tag=f"{window_tag}Plot", label="Widmo czestotliwosci", height=SpectrumPlotHeight, width=SpectrumPlotWidth):
            dpg.add_plot_axis(dpg.mvXAxis, label="Czestotliwosc [Hz]")
            with dpg.plot_axis(dpg.mvYAxis, label="Amplituda") as y_axis:
                line_series_tag = "Spec2"
                if dpg.does_item_exist(line_series_tag):
                    dpg.set_value(line_series_tag, [freqs.tolist(), spectrum.tolist()])
                else:
                    dpg.add_line_series(freqs.tolist(), spectrum.tolist(), label="Widmo", parent=y_axis)
    
        # Add popup context menu to the plot
        with dpg.popup(parent=f"{window_tag}Plot", mousebutton=dpg.mvMouseButton_Right):
            dpg.add_text("Plot Options")
            plot_file_path =  os.path.join(os.path.dirname(audio_file_path), f"{Path(audio_file_path).stem}_SPECTRUM.png")
            dpg.add_button(label="Save Plot", callback=lambda: save_plot_fullscreen(f"{window_tag}Plot", plot_file_path))


def show_original_plot_callback():
    global audio_file_original, sampling_rate_original, audio_file_path_original
    if audio_file_original is None:
        print("Najpierw wczytaj plik!")
        return

    if not isinstance(audio_file_original, np.ndarray):
        print("Blad: dane audio nie sa numpy array!")
        return

    # Sprawdzenie stereo/mono
    if audio_file_original.ndim == 2:
        data_to_plot = audio_file_original[:, 0]
    else:
        data_to_plot = audio_file_original

    if data_to_plot.size == 0:
        print("Brak danych audio do wyswietlenia.")
        return

    # --- DOWN SAMPLING ---
    max_points = 5000
    if len(data_to_plot) <= max_points:
        downsampled_data = data_to_plot
        downsampled_x = np.arange(len(data_to_plot))
    else:
        factor = len(data_to_plot) // max_points
        downsampled_data = data_to_plot[::factor]
        downsampled_x = np.arange(0, len(data_to_plot), factor)

        # Czasami X moze byc dluzszy niz Y przez zaokraglenie — przycinamy
        min_len = min(len(downsampled_data), len(downsampled_x))
        downsampled_data = downsampled_data[:min_len]
        downsampled_x = downsampled_x[:min_len]

    window_tag="OriginalAmplitudeWindow"
    # --- USUWANIE STAREGO OKNA ---
    if dpg.does_item_exist(window_tag):
        dpg.delete_item(window_tag)

    # --- TWORZENIE NOWEGO OKNA Z WYKRESEM ---
    with dpg.window(label="Wykres Amplitudy czystego pliku", tag=window_tag, 
                    on_close=lambda: delete_item(window_tag)):
        apply_window_geometry(window_tag, default_pos=(820, 10), default_size=(500, 250))
        
        with dpg.plot(tag=f"{window_tag}Plot", label="Amplituda w czasie", height=AmplitudePlotHeight, width=AmplitudePlotWidth):
            dpg.add_plot_axis(dpg.mvXAxis, label="Probki")
            with dpg.plot_axis(dpg.mvYAxis, label="Amplituda") as y_axis:
                line_tag = "orig_amp_line"
                dpg.add_line_series([], [], parent=y_axis)
                dpg.add_line_series([], [], parent=y_axis)
                line_series_tag = "Amp0"
                if dpg.does_item_exist(line_series_tag):
                    dpg.set_value(line_series_tag, [downsampled_x.tolist(), downsampled_data.tolist()])
                else:
                    dpg.add_line_series(downsampled_x.tolist(), downsampled_data.tolist(), label="Amplituda", parent=y_axis, tag=line_tag)

        # Add popup context menu to the plot
        with dpg.popup(parent=f"{window_tag}Plot", mousebutton=dpg.mvMouseButton_Right):
            dpg.add_text("Plot Options")
            plot_file_path =  os.path.join(os.path.dirname(audio_file_path_original), f"{Path(audio_file_path_original).stem}_AMPLITUDE.png")
            dpg.add_button(label="Save Plot", callback=lambda: save_plot_fullscreen(f"{window_tag}Plot", plot_file_path))

def show_original_spectrum_callback():
    global audio_file_original, sampling_rate_original, audio_file_path_original
    if audio_file_original is None:
        print("Najpierw wczytaj plik!")
        return

    # Jesli dane sa typu int16, przeskaluj do floatow -1..1 tylko na potrzeby wyswietlenia
    if audio_file_original.dtype == np.int16:
        data_to_plot = audio_file_original.astype(np.float32) / 32767.0
    else:
        data_to_plot = np.copy(audio_file_original)

    if sampling_rate_original is None:
        print("sampling_rate_original is None, setting 44100Hz")
        sampling_rate_original = 44100

    # Obsluga mono/stereo
    if data_to_plot.ndim == 2:
        data_to_plot = data_to_plot[:, 0]  # Wybieramy pierwszy kanal

    if data_to_plot.size == 0:
        print("Brak danych do wyswietlenia.")
        return

    # --- FFT ---
    spectrum = np.abs(np.fft.fft(data_to_plot))
    freqs = np.fft.fftfreq(len(data_to_plot), d=1.0 / sampling_rate_original)

    # Uzywamy tylko dodatnich czestotliwosci
    half = len(freqs) // 2
    freqs = freqs[:half]
    spectrum = spectrum[:half]

    # --- DOWNSAMPLING --- (dla lepszej wydajnosci przy duzych plikach)
    max_points = 5000
    if len(freqs) > max_points:
        factor = len(freqs) // max_points
        spectrum = spectrum[::factor]
        freqs = freqs[::factor]

    # Upewniamy sie, ze X i Y maja te sama dlugosc
    min_len = min(len(freqs), len(spectrum))
    freqs = freqs[:min_len]
    spectrum = spectrum[:min_len]

    window_tag="OriginalSpectrumWindow"
    if dpg.does_item_exist(window_tag):
        dpg.delete_item(window_tag)

    # --- TWORZENIE NOWEGO OKNA Z WYKRESEM ---
    with dpg.window(label="Widmo czestotliwosci (FFT) czystego pliku", tag=window_tag, 
                    on_close=lambda: delete_item(window_tag)):
        apply_window_geometry(window_tag, default_pos=(820, 270), default_size=(500, 250))

        with dpg.plot(tag=f"{window_tag}Plot", label="Widmo czestotliwosci", height=SpectrumPlotHeight, width=SpectrumPlotWidth):
            dpg.add_plot_axis(dpg.mvXAxis, label="Czestotliwosc [Hz]")
            with dpg.plot_axis(dpg.mvYAxis, label="Amplituda") as y_axis:
                line_tag = "orig_spec_line"
                dpg.add_line_series([], [], parent=y_axis)
                dpg.add_line_series([], [], parent=y_axis)
                line_series_tag = "Spec0"
                if dpg.does_item_exist(line_series_tag):
                    dpg.set_value(line_series_tag, [freqs.tolist(), spectrum.tolist()])
                else:
                    dpg.add_line_series(freqs.tolist(), spectrum.tolist(), label="Widmo", parent=y_axis, tag=line_tag)

        # Add popup context menu to the plot
        with dpg.popup(parent=f"{window_tag}Plot", mousebutton=dpg.mvMouseButton_Right):
            dpg.add_text("Plot Options")
            plot_file_path =  os.path.join(os.path.dirname(audio_file_path_original), f"{Path(audio_file_path_original).stem}_SPECTRUM.png")
            dpg.add_button(label="Save Plot", callback=lambda: save_plot_fullscreen(f"{window_tag}Plot", plot_file_path))


AmplitudeFilteredPlotHeight = AmplitudePlotHeight
AmplitudeFilteredPlotWidth = AmplitudePlotWidth
SpectrumFilteredPlotHeight = SpectrumPlotHeight
SpectrumFilteredPlotWidth = SpectrumPlotWidth

def show_filtered_plot_callback():
    global audio_file_filtered, audio_file_path_filtered
    if audio_file_filtered is None:
        print("Najpierw wczytaj plik!")
        return

    if not isinstance(audio_file_filtered, np.ndarray):
        print("Blad: dane audio nie sa numpy array!")
        return

    # Sprawdzenie stereo/mono
    if audio_file_filtered.ndim == 2:
        data_to_plot = audio_file_filtered[:, 0]
    else:
        data_to_plot = audio_file_filtered

    if data_to_plot.size == 0:
        print("Brak danych audio do wyswietlenia.")
        return

    # --- DOWN SAMPLING ---
    max_points = 5000
    if len(data_to_plot) <= max_points:
        downsampled_data = data_to_plot
        downsampled_x = np.arange(len(data_to_plot))
    else:
        factor = len(data_to_plot) // max_points
        downsampled_data = data_to_plot[::factor]
        downsampled_x = np.arange(0, len(data_to_plot), factor)

        # Czasami X moze byc dluzszy niz Y przez zaokraglenie — przycinamy
        min_len = min(len(downsampled_data), len(downsampled_x))
        downsampled_data = downsampled_data[:min_len]
        downsampled_x = downsampled_x[:min_len]

    window_tag = "AmplitudeWindowFiltered"
    # --- USUWANIE STAREGO OKNA ---
    if dpg.does_item_exist(window_tag):
        dpg.delete_item(window_tag)

    # --- TWORZENIE NOWEGO OKNA Z WYKRESEM ---
    with dpg.window(label="Wykres Amplitudy", tag=window_tag, 
                    on_close=lambda: delete_item(window_tag)):
        apply_window_geometry(window_tag, default_pos=(820, 10), default_size=(500, 250))
        with dpg.plot(tag=f"{window_tag}Plot", label="Amplituda w czasie", height=AmplitudePlotHeight, width=AmplitudePlotWidth):
            dpg.add_plot_axis(dpg.mvXAxis, label="Probki")
            with dpg.plot_axis(dpg.mvYAxis, label="Amplituda") as y_axis:
                line_series_tag = "Amp1"
                if dpg.does_item_exist(line_series_tag):
                    dpg.set_value(line_series_tag, [downsampled_x.tolist(), downsampled_data.tolist()])
                else:
                    dpg.add_line_series(downsampled_x.tolist(), downsampled_data.tolist(), label="Amplituda", parent=y_axis)

        # Add popup context menu to the plot
        with dpg.popup(parent=f"{window_tag}Plot", mousebutton=dpg.mvMouseButton_Right):
            dpg.add_text("Plot Options")
            filaname = get_audio_file_path_filtered()
            path = os.path.join(os.path.dirname(filaname), f"{Path(filaname).stem}_AMPLITUDE.png")
            dpg.add_button(label="Save Plot", callback=lambda: save_plot_fullscreen(f"{window_tag}Plot", path))

def show_filtered_spectrum_callback():
    global audio_file_filtered, sampling_rate, audio_file_path_filtered

    if audio_file_filtered is None:
        print("Najpierw wczytaj plik!")
        return

    # Jesli dane sa typu int16, przeskaluj do floatow -1..1 tylko na potrzeby wyswietlenia
    if audio_file_filtered.dtype == np.int16:
        data_to_plot = audio_file_filtered.astype(np.float32) / 32767.0
    else:
        data_to_plot = np.copy(audio_file_filtered)

    # Obsluga mono/stereo
    if data_to_plot.ndim == 2:
        data_to_plot = data_to_plot[:, 0]  # Wybieramy pierwszy kanal

    if data_to_plot.size == 0:
        print("Brak danych do wyswietlenia.")
        return

    # --- FFT ---
    spectrum = np.abs(np.fft.fft(data_to_plot))
    freqs = np.fft.fftfreq(len(data_to_plot), d=1.0 / sampling_rate)

    # Uzywamy tylko dodatnich czestotliwosci
    half = len(freqs) // 2
    freqs = freqs[:half]
    spectrum = spectrum[:half]

    # --- DOWNSAMPLING --- (dla lepszej wydajnosci przy duzych plikach)
    max_points = 5000
    if len(freqs) > max_points:
        factor = len(freqs) // max_points
        spectrum = spectrum[::factor]
        freqs = freqs[::factor]

    # Upewniamy sie, ze X i Y maja te sama dlugosc
    min_len = min(len(freqs), len(spectrum))
    freqs = freqs[:min_len]
    spectrum = spectrum[:min_len]

    window_tag = "SpectrumWindowFiltered"
    # --- USUWANIE STAREGO OKNA ---
    if dpg.does_item_exist(window_tag):
        dpg.delete_item(window_tag)

    # --- TWORZENIE NOWEGO OKNA Z WYKRESEM ---
    with dpg.window(label="Widmo czestotliwosci (FFT) po filtracji", tag=window_tag, 
                    on_close=lambda: delete_item(window_tag)):
        apply_window_geometry(window_tag, default_pos=(820, 270), default_size=(500, 250))

        with dpg.plot(tag=f"{window_tag}Plot", label="Widmo czestotliwosci", height=SpectrumFilteredPlotHeight, width=SpectrumFilteredPlotWidth):
            dpg.add_plot_axis(dpg.mvXAxis, label="Czestotliwosc [Hz]")
            with dpg.plot_axis(dpg.mvYAxis, label="Amplituda") as y_axis:
                line_series_tag = "Spec1"
                if dpg.does_item_exist(line_series_tag):
                    dpg.set_value(line_series_tag, [freqs.tolist(), spectrum.tolist()])
                else:
                    dpg.add_line_series(freqs.tolist(), spectrum.tolist(), label="Widmo", parent=y_axis)

            # Add popup context menu to the plot
            with dpg.popup(parent=f"{window_tag}Plot", mousebutton=dpg.mvMouseButton_Right):
                dpg.add_text("Plot Options")
                plot_file_path =  os.path.join(os.path.dirname(audio_file_path_filtered), f"{Path(audio_file_path_filtered).stem}_SPECTRUM.png")
                dpg.add_button(label="Save Plot", callback=lambda: save_plot_fullscreen(f"{window_tag}Plot", plot_file_path))


BeforeAfterWindowAmplitudeWidth=-1
BeforeAfterWindowAmplitudeHeight=-1
BeforeAfterWindowSpectrumWidth=-1
BeforeAfterWindowSpectrumHeight=-1
DifferenceAmplitudeWindowWidth=-1
DifferenceAmplitudeWindowHeight=-1
DifferenceSpectrumWindowWidth=-1
DifferenceSpectrumWindowHeight=-1
BeforeAfterPlotWindowCollapse=True
DifferenceAmplitudeWindowCollapse=True
BeforeAfterFrequencyDomainWindowCollapse=True
DifferenceFrequencyDomainWindowCollapse=True

def show_signal_difference():
    global audio_file, audio_file_original, audio_file_filtered, sampling_rate, sampling_rate_filtered, \
            BeforeAfterPlotWindowCollapse, \
            DifferenceAmplitudeWindowCollapse, \
            BeforeAfterFrequencyDomainWindowCollapse, \
            DifferenceFrequencyDomainWindowCollapse
    
    if audio_file is None or audio_file_filtered is None:
        print("Brak danych do porownania.")
        return

    if len(audio_file) != len(audio_file_filtered):
        print(f"Sygnaly maja rozne dlugosci:" \
            f"audio_file: {len(audio_file)}," \
            f"audio_file_filtered: {len(audio_file_filtered)},")
        return

    # Downsampling dla plynnosci
    max_points = 10000
    step = max(1, len(audio_file) // max_points)
    x_data = np.arange(0, len(audio_file), step)
    y_noised = audio_file[::step]
    y_filt = audio_file_filtered[::step]

    # --- Roznica w dziedzinie czasu
    diff_signal = np.array(audio_file) - np.array(audio_file_filtered)
    max_points = 10000
    step = max(1, len(diff_signal) // max_points)
    x_data = np.arange(0, len(diff_signal), step)
    y_diff = diff_signal[::step]

    # --- Roznica w dziedzinie czestotliwosci
    fft_noised = np.fft.fft(audio_file)
    fft_filt = np.fft.fft(audio_file_filtered)
    fft_diff_noised = np.abs(fft_noised) - np.abs(fft_filt) 
    freqs = np.fft.fftfreq(len(audio_file), 1 / sampling_rate)
    
    half_n = len(freqs) // 2
    freqs = freqs[:half_n]
    fft_diff_noised = fft_diff_noised[:half_n]

    print(len(freqs))
    print(len(fft_noised))
    print(len(fft_filt))
    print(len(fft_diff_noised))

    # half_n = min(len(freqs), len(fft_orig) // 2, len(fft_filt) // 2)
    # freqs_half = freqs[:half_n]
    fft_noised_half = np.abs(fft_noised[:half_n])
    fft_filt_half = np.abs(fft_filt[:half_n])

    y_original = None
    y_diff_original = None
    fft_original_half = None
    fft_diff_original = None


    if audio_file_original is not None and len(audio_file_original) == len(audio_file): 
        y_original = audio_file_original[::step]
        
        diff_signal_original = np.array(audio_file_original) - np.array(audio_file_filtered)
        y_diff_original = diff_signal_original[::step]

        fft_original = np.fft.fft(audio_file_original)
        fft_diff_original = np.abs(fft_original) - np.abs(fft_filt) 
        fft_original_half = np.abs(fft_original[:half_n])

    with dpg.theme() as custom_line_theme:
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, [0, 255, 0, 80], category=dpg.mvThemeCat_Plots) # Green with alpha = 100

    window_tag_ABA = "BeforeAfterPlotWindow"
    if dpg.does_item_exist(window_tag_ABA):
        dpg.delete_item(window_tag_ABA)

    with dpg.window(label="Amplituda sygnalu", collapsed=BeforeAfterPlotWindowCollapse, tag=window_tag_ABA, width=BeforeAfterWindowAmplitudeWidth, height=BeforeAfterWindowAmplitudeHeight, 
                    on_close=lambda: delete_item(window_tag_ABA)):
        apply_window_geometry(window_tag_ABA, default_pos=(820, 10), default_size=(600, 300))

        with dpg.plot(tag=f"{window_tag_ABA}Plot", parent=window_tag_ABA, height=BeforeAfterWindowAmplitudeHeight, width=BeforeAfterWindowAmplitudeWidth):
            dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthEast)
            dpg.add_plot_axis(dpg.mvXAxis, label="Numer probki")
            with dpg.plot_axis(dpg.mvYAxis, label="Amplituda") as y_axis:
                dpg.add_line_series(x_data.tolist(), y_noised.tolist(), label="Zaszumiony", parent=y_axis)
                dpg.add_line_series(x_data.tolist(), y_filt.tolist(), label="Po filtracji", parent=y_axis)
                if y_original is not None:
                    line_series_tag  = dpg.add_line_series(x_data.tolist(), y_original.tolist(), label="Oryginalny", parent=y_axis)
                    dpg.set_item_theme(line_series_tag, custom_line_theme)


        # Add popup context menu to the plot
        with dpg.popup(parent=f"{window_tag_ABA}Plot", mousebutton=dpg.mvMouseButton_Right):
            dpg.add_text("Plot Options")
            plot_file_path_ac =  os.path.join(os.path.dirname(audio_file_path), f"{Path(audio_file_path).stem}_AMPLITUDE_COMPARE.png")
            dpg.add_button(label="Save Plot", callback=lambda: save_plot_fullscreen(f"{window_tag_ABA}Plot", plot_file_path_ac))


    window_tag_ADiff = "DifferenceAmplitudeWindow"
    if dpg.does_item_exist(window_tag_ADiff):
        dpg.delete_item(window_tag_ADiff)

    with dpg.window(label="Roznica amplitudy", collapsed=DifferenceAmplitudeWindowCollapse, tag=window_tag_ADiff, width=DifferenceAmplitudeWindowWidth, height=DifferenceAmplitudeWindowHeight, 
                    on_close=lambda: delete_item(window_tag_ADiff)):
        apply_window_geometry(window_tag_ADiff, default_pos=(820, 330), default_size=(600, 300))

        with dpg.plot(tag=f"{window_tag_ADiff}Plot", height=DifferenceAmplitudeWindowHeight, width=DifferenceAmplitudeWindowWidth):
            dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthEast)
            dpg.add_plot_axis(dpg.mvXAxis, label="Numer probki")
            with dpg.plot_axis(dpg.mvYAxis, label="Roznica amplitud") as y_axis:
                dpg.add_line_series(x_data.tolist(), y_diff.tolist(), label="Roznica N-F", parent=y_axis)
                if y_diff_original is not None:
                    line_series_tag = dpg.add_line_series(x_data.tolist(), y_diff_original.tolist(), label="Roznica O-F", parent=y_axis)
                    dpg.set_item_theme(line_series_tag, custom_line_theme)

            # Add popup context menu to the plot
            with dpg.popup(parent=f"{window_tag_ADiff}Plot", mousebutton=dpg.mvMouseButton_Right):
                dpg.add_text("Plot Options")
                plot_file_path_ad =  os.path.join(os.path.dirname(audio_file_path), f"{Path(audio_file_path).stem}_AMPLITUDE_DIFF.png")
                dpg.add_button(label="Save Plot", callback=lambda: save_plot_fullscreen(f"{window_tag_ADiff}Plot", plot_file_path_ad))


    window_tag_FBA = "BeforeAfterFrequencyDomainWindow"
    if dpg.does_item_exist(window_tag_FBA):
        dpg.delete_item(window_tag_FBA)

    with dpg.window(label="Widmo sygnalu", collapsed=BeforeAfterFrequencyDomainWindowCollapse, tag=window_tag_FBA, width=BeforeAfterWindowAmplitudeWidth, height=BeforeAfterWindowAmplitudeHeight, 
                    on_close=lambda: delete_item(window_tag_FBA)):
        apply_window_geometry(window_tag_FBA, default_pos=(820, 10), default_size=(600, 300))

        with dpg.plot(tag=f"{window_tag_FBA}Plot", height=BeforeAfterWindowAmplitudeHeight, width=BeforeAfterWindowAmplitudeWidth):
            dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthEast)
            dpg.add_plot_axis(dpg.mvXAxis, label="Czestotliwosc [Hz]")
            with dpg.plot_axis(dpg.mvYAxis, label="Amplituda") as y_axis:
                dpg.add_line_series(freqs.tolist(), fft_noised_half.tolist(), label="Zaszumione widmo", parent=y_axis)
                dpg.add_line_series(freqs.tolist(), fft_filt_half.tolist(), label="Widmo po filtracji", parent=y_axis)
                if fft_original_half is not None:
                    line_series_tag = dpg.add_line_series(freqs.tolist(), fft_original_half.tolist(), label="Oryginalne widmo", parent=y_axis)
                    dpg.set_item_theme(line_series_tag, custom_line_theme)

        # Add popup context menu to the plot
        with dpg.popup(parent=f"{window_tag_FBA}Plot", mousebutton=dpg.mvMouseButton_Right):
            dpg.add_text("Plot Options")
            plot_file_path_sc =  os.path.join(os.path.dirname(audio_file_path), f"{Path(audio_file_path).stem}_SPECTRUM_COMPARE.png")
            dpg.add_button(label="Save Plot", callback=lambda: save_plot_fullscreen(f"{window_tag_FBA}Plot", plot_file_path_sc))


    window_tag_FDiff = "DifferenceFrequencyDomainWindow"
    if dpg.does_item_exist(window_tag_FDiff):
        dpg.delete_item(window_tag_FDiff)

    with dpg.window(label="Roznica widm", collapsed=DifferenceFrequencyDomainWindowCollapse, tag=window_tag_FDiff, width=DifferenceSpectrumWindowWidth, height=DifferenceSpectrumWindowHeight, 
                    on_close=lambda: delete_item(window_tag_FDiff)):
        apply_window_geometry(window_tag_FDiff, default_pos=(820, 660), default_size=(600, 300))

        with dpg.plot(tag=f"{window_tag_FDiff}Plot", height=DifferenceSpectrumWindowHeight, width=DifferenceSpectrumWindowWidth):
            dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthEast)
            dpg.add_plot_axis(dpg.mvXAxis, label="Czestotliwosc [Hz]")
            with dpg.plot_axis(dpg.mvYAxis, label="Roznica widm") as y_axis:
                dpg.add_line_series(freqs.tolist(), fft_diff_noised.tolist(), label="N - F", parent=y_axis)
                if fft_diff_original is not None:
                    line_series_tag = dpg.add_line_series(freqs.tolist(), fft_diff_original.tolist(), label="O - F", parent=y_axis)
                    dpg.set_item_theme(line_series_tag, custom_line_theme)

        # Add popup context menu to the plot
        with dpg.popup(parent=f"{window_tag_FDiff}Plot", mousebutton=dpg.mvMouseButton_Right):
            dpg.add_text("Plot Options")
            plot_file_path_sd =  os.path.join(os.path.dirname(audio_file_path), f"{Path(audio_file_path).stem}_SPECTRUM_DIFF.png")
            dpg.add_button(label="Save Plot", callback=lambda: save_plot_fullscreen(f"{window_tag_FDiff}Plot", plot_file_path_sd))


def save_plot_fullscreen(plot_tag: str, filename="saved_plot.png"):
    print(f"Saving as: {filename}")

    # Find parent window of the plot
    plot_info = dpg.get_item_info(plot_tag)
    if plot_info is None:
        print(f"Plot tag {plot_tag} not found!")
        return
    
    parent = dpg.get_item_parent(plot_tag)
    window_tag = None
    
    try:
        while parent:
            parent_info = dpg.get_item_info(parent)
            if parent_info is None:
                print(f"Parent info is None for {parent}")
                break
            
            print(f"Parent {parent} type: {parent_info.get('type')}")
            
            if parent_info.get("type") == "mvAppItemType::mvWindowAppItem":
                window_tag = parent
                break
            
            parent = parent_info.get("parent")
        
        if window_tag is None:
            print("Plot is not inside a window!")
            return

        # Check if the window exists and is visible
        if not dpg.does_item_exist(window_tag):
            print(f"Window {window_tag} no longer exists.")
            return

        #focus window to bring it to the top    
        dpg.focus_item(window_tag)

        # Save original window size and pos
        orig_pos = dpg.get_item_pos(window_tag)
        orig_size = dpg.get_item_width(window_tag), dpg.get_item_height(window_tag)

        # Get viewport size (fullscreen)
        vp_width = dpg.get_viewport_width()
        vp_height = dpg.get_viewport_height() - 30
        
        # Resize window to fullscreen viewport
        dpg.configure_item(window_tag, pos=(0, 0), width=vp_width, height=vp_height)
        # Need to force render update so framebuffer captures updated window
        dpg.render_dearpygui_frame()
        time.sleep(0.5)
        # Output framebuffer (captures the whole window)
        dpg.output_frame_buffer(filename)
        # time.sleep(0.5)
        # dpg.render_dearpygui_frame()
        
        # Restore window size and position
        dpg.configure_item(window_tag, pos=orig_pos, width=orig_size[0], height=orig_size[1])
        
        print(f"Saved plot image to {filename}")
    except Exception as e:
        print(f"ERROR while generating plot: {e}")


###############################################################################################################################################################

#  __         ______   __       __  _______    ______    ______    ______  
# /  |       /      \ /  |  _  /  |/       \  /      \  /      \  /      \ 
# $$ |      /$$$$$$  |$$ | / \ $$ |$$$$$$$  |/$$$$$$  |/$$$$$$  |/$$$$$$  |
# $$ |      $$ |  $$ |$$ |/$  \$$ |$$ |__$$ |$$ |__$$ |$$ \__$$/ $$ \__$$/ 
# $$ |      $$ |  $$ |$$ /$$$  $$ |$$    $$/ $$    $$ |$$      \ $$      \ 
# $$ |      $$ |  $$ |$$ $$/$$ $$ |$$$$$$$/  $$$$$$$$ | $$$$$$  | $$$$$$  |
# $$ |_____ $$ \__$$ |$$$$/  $$$$ |$$ |      $$ |  $$ |/  \__$$ |/  \__$$ |
# $$       |$$    $$/ $$$/    $$$ |$$ |      $$ |  $$ |$$    $$/ $$    $$/ 
# $$$$$$$$/  $$$$$$/  $$/      $$/ $$/       $$/   $$/  $$$$$$/   $$$$$$/  

###############################################################################################################################################################

def open_LOWPASS_filter_popup_callback():
    dpg.show_item("lowpass_filter_popup")

def apply_LOWPASS_filter_callback():
    global audio_file, audio_file_filtered, sampling_rate, audio_file_path_filtered, sampling_rate_filtered

    if audio_file is None:
        print("Najpierw wczytaj plik!")
        return

    prepare_environment_for_filtering()

    filter_type = dpg.get_value("low_filter_type")
    filter_order = int(dpg.get_value("low_filter_order"))
    cutoff_freq = int(dpg.get_value("low_cutoff_freq"))
    sampling_rate_filtered = int(dpg.get_value("low_sampling_rate"))

    if cutoff_freq <= 0 or cutoff_freq >= 0.5 * sampling_rate_filtered:
        print("Niepoprawna czestotliwosc odciecia!")
        return

    nyquist = 0.5 * sampling_rate_filtered
    normalized_cutoff = cutoff_freq / nyquist

    start_benchmark()
    start_processing_indicator()

    # Wybor filtra
    if filter_type == "Butterworth":
        b, a = butter(filter_order, normalized_cutoff, btype='low', analog=False)
    elif filter_type == "Chebyshev":
        b, a = cheby1(filter_order, 1, normalized_cutoff, btype='low', analog=False)
    elif filter_type == "Bessel":
        b, a = bessel(filter_order, normalized_cutoff, btype='low', analog=False, norm='phase')
    else:
        print("Nieznany typ filtra!")
        return

    # # Filtrujemy kazdy kanal osobno
    if audio_file.ndim == 2:
        for channel in range(audio_file.shape[1]):
            audio_file_filtered[:, channel] = filtfilt(b, a, audio_file[:, channel])
    else:
        audio_file_filtered = filtfilt(b, a, audio_file)

    stop_benchmark_and_show_results()
    stop_processing_indicator()

    thd_before = calculate_thd(audio_file, sampling_rate)
    thd_low = calculate_thd(audio_file_filtered, sampling_rate_filtered)

    file_original_path = Path(audio_file_path)
    audio_file_path_filtered =  os.path.join(os.path.dirname(audio_file_path), file_original_path.with_stem(f"{Path(audio_file_path).stem}_LOW_{filter_type}_{filter_order}_{cutoff_freq}"))
    
    print(f"Zastosowano filtr: {filter_type}, rzad: {filter_order}, odciecie: {cutoff_freq} Hz, THD before: {thd_before} THD after: {thd_low}")

    show_filtered_plot_callback()
    show_filtered_spectrum_callback()
    call_create_audio_filtered_play_callback()

    print(f"Zapisywanie do pliku: {audio_file_path_filtered}")
    save_audio_with_convert(audio_file_path_filtered, sampling_rate_filtered, audio_file_filtered)
    print(f"Zapisano do pliku: {audio_file_path_filtered}")
    dpg.hide_item("lowpass_filter_popup")

# Okno Popup Filtracji
with dpg.window(label="Filtracja dolnoprzepustowa", tag="lowpass_filter_popup", modal=True, show=False, width=800, height=200):
    dpg.add_combo(("Butterworth", "Chebyshev", "Bessel"), label="Typ filtra", tag="low_filter_type", default_value="Butterworth")
    dpg.add_input_int(label="Rzad filtra", tag="low_filter_order", default_value=4, min_value=1, max_value=10)
    dpg.add_input_int(label="Czestotliwosc odciecia (Hz)", tag="low_cutoff_freq", default_value=5000, min_value=1, max_value=22000)
    dpg.add_input_int(label="Czestotliwosc probkowania (Hz)", tag="low_sampling_rate", default_value=44100, min_value=1000, max_value=96000)
    dpg.add_button(label="Zastosuj filtr", callback=apply_LOWPASS_filter_callback)


###############################################################################################################################################################

#  __    __  ______   ______   __    __  _______    ______    ______    ______  
# /  |  /  |/      | /      \ /  |  /  |/       \  /      \  /      \  /      \ 
# $$ |  $$ |$$$$$$/ /$$$$$$  |$$ |  $$ |$$$$$$$  |/$$$$$$  |/$$$$$$  |/$$$$$$  |
# $$ |__$$ |  $$ |  $$ | _$$/ $$ |__$$ |$$ |__$$ |$$ |__$$ |$$ \__$$/ $$ \__$$/ 
# $$    $$ |  $$ |  $$ |/    |$$    $$ |$$    $$/ $$    $$ |$$      \ $$      \ 
# $$$$$$$$ |  $$ |  $$ |$$$$ |$$$$$$$$ |$$$$$$$/  $$$$$$$$ | $$$$$$  | $$$$$$  |
# $$ |  $$ | _$$ |_ $$ \__$$ |$$ |  $$ |$$ |      $$ |  $$ |/  \__$$ |/  \__$$ |
# $$ |  $$ |/ $$   |$$    $$/ $$ |  $$ |$$ |      $$ |  $$ |$$    $$/ $$    $$/ 
# $$/   $$/ $$$$$$/  $$$$$$/  $$/   $$/ $$/       $$/   $$/  $$$$$$/   $$$$$$/  

###############################################################################################################################################################

def open_HIPASS_filter_popup_callback():
    dpg.show_item("hipass_filter_popup")

def apply_HIPASS_filter_callback():
    global audio_file, audio_file_filtered, sampling_rate, audio_file_path_filtered, sampling_rate_filtered

    if audio_file is None:
        print("Najpierw wczytaj plik!")
        return

    prepare_environment_for_filtering()

    filter_type = dpg.get_value("hi_filter_type")
    filter_order = int(dpg.get_value("hi_filter_order"))
    cutoff_freq = int(dpg.get_value("hi_cutoff_freq"))
    sampling_rate_filtered = int(dpg.get_value("hi_sampling_rate"))

    if cutoff_freq <= 0 or cutoff_freq >= 0.5 * sampling_rate_filtered:
        print("Niepoprawna czestotliwosc odciecia!")
        return

    nyquist = 0.5 * sampling_rate_filtered
    normalized_cutoff = cutoff_freq / nyquist

    start_benchmark()
    start_processing_indicator()

    # Wybor filtra
    if filter_type == "Butterworth":
        b, a = butter(filter_order, normalized_cutoff, btype='high', analog=False)
    elif filter_type == "Chebyshev":
        b, a = cheby1(filter_order, 1, normalized_cutoff, btype='high', analog=False)
    elif filter_type == "Bessel":
        b, a = bessel(filter_order, normalized_cutoff, btype='high', analog=False, norm='phase')
    else:
        print("Nieznany typ filtra!")
        return

    # # Filtrujemy kazdy kanal osobno
    if audio_file.ndim == 2:
        for channel in range(audio_file.shape[1]):
            audio_file_filtered[:, channel] = filtfilt(b, a, audio_file[:, channel])
    else:
        audio_file_filtered = filtfilt(b, a, audio_file)

    stop_benchmark_and_show_results()
    stop_processing_indicator()

    thd_before = calculate_thd(audio_file, sampling_rate)
    thd_high = calculate_thd(audio_file_filtered, sampling_rate_filtered)

    print(f"Zastosowano filtr: {filter_type}, rzad: {filter_order}, odciecie: {cutoff_freq} Hz, THD before: {thd_before}, THD: {thd_high}")

    file_original_path = Path(audio_file_path)
    audio_file_path_filtered =  os.path.join(os.path.dirname(audio_file_path), file_original_path.with_stem(f"{Path(audio_file_path).stem}_HIGH_{filter_type}_{filter_order}_{cutoff_freq}"))
    
    show_filtered_plot_callback()
    show_filtered_spectrum_callback()
    call_create_audio_filtered_play_callback()

    print(f"Zapisywanie do pliku: {audio_file_path_filtered}")
    save_audio_with_convert(audio_file_path_filtered, sampling_rate_filtered, audio_file_filtered)
    print(f"Zapisano do pliku: {audio_file_path_filtered}")
    dpg.hide_item("hipass_filter_popup")

# Okno Popup Filtracji
with dpg.window(label="Filtracja gornoprzepustowa", tag="hipass_filter_popup", modal=True, show=False, width=800, height=200):
    dpg.add_combo(("Butterworth", "Chebyshev", "Bessel"), label="Typ filtra", tag="hi_filter_type", default_value="Butterworth")
    dpg.add_input_int(label="Rzad filtra", tag="hi_filter_order", default_value=4, min_value=1, max_value=10)
    dpg.add_input_int(label="Czestotliwosc odciecia (Hz)", tag="hi_cutoff_freq", default_value=5000, min_value=1, max_value=22000)
    dpg.add_input_int(label="Czestotliwosc probkowania (Hz)", tag="hi_sampling_rate", default_value=44100, min_value=1000, max_value=96000)
    dpg.add_button(label="Zastosuj filtr", callback=apply_HIPASS_filter_callback)


###############################################################################################################################################################

#  _______    ______   __    __  _______   _______    ______    ______    ______  
# /       \  /      \ /  \  /  |/       \ /       \  /      \  /      \  /      \ 
# $$$$$$$  |/$$$$$$  |$$  \ $$ |$$$$$$$  |$$$$$$$  |/$$$$$$  |/$$$$$$  |/$$$$$$  |
# $$ |__$$ |$$ |__$$ |$$$  \$$ |$$ |  $$ |$$ |__$$ |$$ |__$$ |$$ \__$$/ $$ \__$$/ 
# $$    $$< $$    $$ |$$$$  $$ |$$ |  $$ |$$    $$/ $$    $$ |$$      \ $$      \ 
# $$$$$$$  |$$$$$$$$ |$$ $$ $$ |$$ |  $$ |$$$$$$$/  $$$$$$$$ | $$$$$$  | $$$$$$  |
# $$ |__$$ |$$ |  $$ |$$ |$$$$ |$$ |__$$ |$$ |      $$ |  $$ |/  \__$$ |/  \__$$ |
# $$    $$/ $$ |  $$ |$$ | $$$ |$$    $$/ $$ |      $$ |  $$ |$$    $$/ $$    $$/ 
# $$$$$$$/  $$/   $$/ $$/   $$/ $$$$$$$/  $$/       $$/   $$/  $$$$$$/   $$$$$$/  
                                                                                
###############################################################################################################################################################

def open_BANDPASS_filter_popup_callback():
    dpg.show_item("bandpass_filter_popup")

def apply_BANDPASS_filter_callback():
    global audio_file, audio_file_filtered, sampling_rate, audio_file_path_filtered, sampling_rate_filtered

    if audio_file is None:
        print("Najpierw wczytaj plik!")
        return

    prepare_environment_for_filtering()

    filter_type = dpg.get_value("band_filter_type")
    filter_order = int(dpg.get_value("band_filter_order"))
    cutoff_low = int(dpg.get_value("band_cutoff_freq_low"))
    cutoff_high = int(dpg.get_value("band_cutoff_freq_high"))
    sampling_rate_filtered = int(dpg.get_value("band_sampling_rate"))

    if cutoff_low <= 0 or cutoff_high <= cutoff_low or cutoff_high >= sampling_rate_filtered / 2:
        print("Niepoprawne czestotliwosci odciecia!")
        return

    nyquist = sampling_rate_filtered / 2
    normalized_cutoff = [cutoff_low / nyquist, cutoff_high / nyquist]

    start_benchmark()
    start_processing_indicator()

    # Wybor filtra
    if filter_type == "Butterworth":
        b, a = butter(filter_order, normalized_cutoff, btype='band', analog=False)
    elif filter_type == "Chebyshev":
        b, a = cheby1(filter_order, 1, normalized_cutoff, btype='band', analog=False)
    elif filter_type == "Bessel":
        b, a = bessel(filter_order, normalized_cutoff, btype='band', analog=False, norm='phase')
    else:
        print("Nieznany typ filtra!")
        return

    # Filtrujemy kazdy kanal osobno
    if audio_file.ndim == 2:
        for channel in range(audio_file.shape[1]):
            audio_file_filtered[:, channel] = filtfilt(b, a, audio_file[:, channel])
    else:
        audio_file_filtered = filtfilt(b, a, audio_file)

    stop_benchmark_and_show_results()
    stop_processing_indicator()

    thd_before = calculate_thd(audio_file, sampling_rate) 
    thd_band = calculate_thd(audio_file_filtered, sampling_rate_filtered)

    print(f"Zastosowano filtr: {filter_type}, rzad: {filter_order}, odciecie dolne: {cutoff_low}, odciecie gorne: {cutoff_high} Hz, THD before: {thd_before}, THD: {thd_band}")

    file_original_path = Path(audio_file_path)
    audio_file_path_filtered =  os.path.join(os.path.dirname(audio_file_path), file_original_path.with_stem(f"{Path(audio_file_path).stem}_BAND_{filter_type}_{filter_order}_{cutoff_low}-{cutoff_high}"))
    
    show_filtered_plot_callback()
    show_filtered_spectrum_callback()
    call_create_audio_filtered_play_callback()

    print(f"Zapisywanie do pliku: {audio_file_path_filtered}")
    save_audio_with_convert(audio_file_path_filtered, sampling_rate_filtered, audio_file_filtered)
    print(f"Zapisano do pliku: {audio_file_path_filtered}")
    dpg.hide_item("bandpass_filter_popup")

# Okno Popup Filtracji
with dpg.window(label="Filtracja pasmowoprzepustowa", tag="bandpass_filter_popup", modal=True, show=False, width=800, height=250):
    dpg.add_combo(("Butterworth", "Chebyshev", "Bessel"), label="Typ filtra", tag="band_filter_type", default_value="Butterworth")
    dpg.add_input_int(label="Rzad filtra", tag="band_filter_order", default_value=4, min_value=1, max_value=10)
    dpg.add_input_int(label="Dolna czestotliwosc odciecia (Hz)", tag="band_cutoff_freq_low", default_value=300, min_value=1, max_value=22000)
    dpg.add_input_int(label="Gorna czestotliwosc odciecia (Hz)", tag="band_cutoff_freq_high", default_value=5000, min_value=1, max_value=22000)
    dpg.add_input_int(label="Czestotliwosc probkowania (Hz)", tag="band_sampling_rate", default_value=44100, min_value=1000, max_value=96000)
    dpg.add_button(label="Zastosuj filtr", callback=apply_BANDPASS_filter_callback)


###############################################################################################################################################################

#  __        __       __   ______  
# /  |      /  \     /  | /      \ 
# $$ |      $$  \   /$$ |/$$$$$$  |
# $$ |      $$$  \ /$$$ |$$ \__$$/ 
# $$ |      $$$$  /$$$$ |$$      \ 
# $$ |      $$ $$ $$/$$ | $$$$$$  |
# $$ |_____ $$ |$$$/ $$ |/  \__$$ |
# $$       |$$ | $/  $$ |$$    $$/ 
# $$$$$$$$/ $$/      $$/  $$$$$$/  

###############################################################################################################################################################

# def lms_filter(x, d, filter_order=32, mu=0.001):
#     """
#     x - input signal
#     d - reference signal
#     filter_order - number of filter coefficients
#     mu - learning rate
#     """
#     try:
#         x = np.asarray(x, dtype=float)
#         d = np.asarray(d, dtype=float)
#     except Exception as e:
#         raise ValueError(f"Unable to convert x or d to numpy array: {e}")

#     if len(x) < filter_order or len(d) < filter_order:
#         raise ValueError("Input signals must be at least as long as the filter_order.")

#     try:
#         x_matrix = pa.input_from_history(x, filter_order)
#         d_vector = d[filter_order-1:]  # Trimming to align dimensions
#     except Exception as e:
#         raise ValueError(f"Error preparing data for LMS: {e}")

#     f = pa.filters.FilterLMS(n=filter_order, mu=mu, w="zeros")
#     y, e, w = f.run(d_vector, x_matrix)

#     return y, e, w

def lms_filter(input_signal, desired, filter_length=32, mu=0.01):
    """
    Implementacja filtra LMS.

    Parameters:
        input_signal (np.ndarray): Sygnał wejściowy (np. z zakłóceniami).
        desired (np.ndarray): Sygnał pożądany (referencyjny).
        filter_length (int): Liczba współczynników filtra (rząd filtra).
        mu (float): Współczynnik uczenia (krok adaptacji).

    Returns:
        output (np.ndarray): Sygnał wyjściowy po filtracji LMS.
        error (np.ndarray): Błąd adaptacji (różnica między desired a output).
        weights (np.ndarray): Ewolucja współczynników filtra w czasie.
    """
    n_samples = len(input_signal)
    weights = np.zeros(filter_length)
    output = np.zeros(n_samples)
    error = np.zeros(n_samples)

    for n in range(filter_length, n_samples):
        x = input_signal[n-filter_length:n][::-1]  # wektor wejściowy (ostatnie próbki, odwrócone)
        y = np.dot(weights, x)                     # sygnał wyjściowy
        output[n] = y
        error[n] = desired[n] - y                  # błąd
        weights += 2 * mu * error[n] * x           # aktualizacja współczynników

    return output, error, weights



def measure_lms_convergence(e, stable_duration=10, threshold_ratio=1.05):
    """
    e - wektor bledu z LMS
    stable_duration - liczba kolejnych probek, przez ktore blad musi byc stabilny
    threshold_ratio - jak blisko musi byc do minimalnego bledu
    """
    mse = e ** 2
    min_mse = np.min(mse)
    threshold = threshold_ratio * min_mse

    for i in range(len(mse) - stable_duration):
        window = mse[i:i+stable_duration]
        if np.all(window < threshold):
            return i  # indeks konwergencji

    return -1  # brak konwergencji

def open_LMS_filter_popup_callback():
    dpg.show_item("lms_filter_popup")

def apply_LMS_filter_callback():
    global audio_file, audio_file_filtered, audio_file_path_filtered, sampling_rate_filtered

    if audio_file is None:
        print("Najpierw wczytaj plik!")
        return

    prepare_environment_for_filtering()

    filter_length = int(dpg.get_value("lms_filter_length"))
    mu = float(dpg.get_value("lms_learning_rate"))
    sampling_rate_filtered = sampling_rate

    start_benchmark()
    start_processing_indicator()

    if audio_file.ndim == 2:
        for channel in range(audio_file.shape[1]):
            audio_file_filtered[:, channel], error, weights = lms_filter(audio_file[:, channel], audio_file[:, channel], filter_length, mu)
    else:
        audio_file_filtered, error, weights  = lms_filter(audio_file, audio_file, filter_length, mu)


    convergence_index = measure_lms_convergence(error)
    print(f"Szybkosc konwergencji: {convergence_index} probek")

    max_len = max(len(audio_file), len(audio_file_filtered))
    audio_file_filtered = np.pad(audio_file_filtered, (max_len - len(audio_file_filtered), 0), mode='constant')

    stop_benchmark_and_show_results()
    stop_processing_indicator()

    file_original_path = Path(audio_file_path)
    audio_file_path_filtered =  os.path.join(os.path.dirname(audio_file_path), file_original_path.with_stem(f"{Path(audio_file_path).stem}_LMS_{filter_length}_{mu}"))
    
    show_filtered_plot_callback()
    show_filtered_spectrum_callback()
    call_create_audio_filtered_play_callback()

    print(f"Zastosowano filtr LMS: dlugosc = {filter_length}, wspolczynnik uczenia = {mu}")
    print(f"Zapisywanie do pliku: {audio_file_path_filtered}")
    save_audio_with_convert(audio_file_path_filtered, sampling_rate_filtered, audio_file_filtered)
    print(f"Zapisano do pliku: {audio_file_path_filtered}")
    dpg.hide_item("lms_filter_popup")

# Okno Popup Filtracji
with dpg.window(label="Filtracja LMS", tag="lms_filter_popup", modal=True, show=False, width=800, height=300):
    dpg.add_input_int(label="Dlugosc filtra", tag="lms_filter_length", default_value=32, min_value=1, step=1, max_value=1024)
    dpg.add_input_float(label="Wspolczynnik uczenia (μ)", tag="lms_learning_rate", default_value=0.0010000, step=0.000001, min_value=0.000001, max_value=1.0)
    dpg.add_button(label="Zastosuj filtr LMS", callback=apply_LMS_filter_callback)



###############################################################################################################################################################

#  __       __   ______   __     __  ________  __        ________  ________  ______  
# /  |  _  /  | /      \ /  |   /  |/        |/  |      /        |/        |/      \ 
# $$ | / \ $$ |/$$$$$$  |$$ |   $$ |$$$$$$$$/ $$ |      $$$$$$$$/ $$$$$$$$//$$$$$$  |
# $$ |/$  \$$ |$$ |__$$ |$$ |   $$ |$$ |__    $$ |      $$ |__       $$ |  $$ \__$$/ 
# $$ /$$$  $$ |$$    $$ |$$  \ /$$/ $$    |   $$ |      $$    |      $$ |  $$      \ 
# $$ $$/$$ $$ |$$$$$$$$ | $$  /$$/  $$$$$/    $$ |      $$$$$/       $$ |   $$$$$$  |
# $$$$/  $$$$ |$$ |  $$ |  $$ $$/   $$ |_____ $$ |_____ $$ |_____    $$ |  /  \__$$ |
# $$$/    $$$ |$$ |  $$ |   $$$/    $$       |$$       |$$       |   $$ |  $$    $$/ 
# $$/      $$/ $$/   $$/     $/     $$$$$$$$/ $$$$$$$$/ $$$$$$$$/    $$/    $$$$$$/  
                                                                                   
###############################################################################################################################################################

def wavelet_filter(signal, wavelet_name='db4', level=4, threshold=0.04):
    coeffs = pywt.wavedec(signal, wavelet_name, level=level)
    
    # Thresholding: zerowanie malych wspolczynnikow
    new_coeffs = []
    for c in coeffs:
        c = pywt.threshold(c, threshold * np.max(c), mode='soft')
        new_coeffs.append(c)
    
    reconstructed_signal = pywt.waverec(new_coeffs, wavelet_name)

    # Dopasuj dlugosc sygnalu
    return reconstructed_signal[:len(signal)]

def open_WAVELET_filter_popup_callback():
    dpg.show_item("wavelet_filter_popup")

def apply_WAVELET_filter_callback():
    global audio_file, audio_file_filtered, audio_file_path_filtered, sampling_rate_filtered

    if audio_file is None:
        print("Najpierw wczytaj plik!")
        return

    prepare_environment_for_filtering()

    wavelet_name = dpg.get_value("wavelet_name")
    decomposition_level = int(dpg.get_value("wavelet_level"))
    threshold = float(dpg.get_value("wavelet_threshold"))
    sampling_rate_filtered = sampling_rate

    start_benchmark()
    start_processing_indicator()

    if audio_file.ndim == 2:
        for channel in range(audio_file.shape[1]):
            audio_file_filtered[:, channel] = wavelet_filter(audio_file[:, channel], wavelet_name, decomposition_level, threshold)
    else:
        audio_file_filtered = wavelet_filter(audio_file, wavelet_name, decomposition_level, threshold)

    stop_benchmark_and_show_results()
    stop_processing_indicator()

    file_original_path = Path(audio_file_path)
    audio_file_path_filtered =  os.path.join(os.path.dirname(audio_file_path), file_original_path.with_stem(f"{Path(audio_file_path).stem}_WAVELET_{wavelet_name}_{decomposition_level}_{threshold}"))
        
    show_filtered_plot_callback()
    show_filtered_spectrum_callback()
    call_create_audio_filtered_play_callback()

    print(f"Zastosowano filtracje wavelet: {wavelet_name}, poziom: {decomposition_level}, prog: {threshold}")
    print(f"Zapisywanie do pliku: {audio_file_path_filtered}")
    save_audio_with_convert(audio_file_path_filtered, sampling_rate_filtered, audio_file_filtered)
    print(f"Zapisano do pliku: {audio_file_path_filtered}")
    dpg.hide_item("wavelet_filter_popup")

# Okno Popup Filtracji
with dpg.window(label="Filtracja Wavelet", tag="wavelet_filter_popup", modal=True, show=False, width=800, height=400):
    dpg.add_combo(("db4", "haar", "sym5", "coif1"), label="Typ falki", tag="wavelet_name", default_value="db4")
    dpg.add_input_int(label="Poziom dekompozycji", tag="wavelet_level", default_value=4, step=1, min_value=1, max_value=10)
    dpg.add_input_float(label="Prog (threshold)", tag="wavelet_threshold", default_value=0.04, step=0.01, min_value=0.001, max_value=1.0)
    dpg.add_button(label="Zastosuj filtr Wavelet", callback=apply_WAVELET_filter_callback)


###############################################################################################################################################################

#  __    __   ______   __        __       __   ______   __    __ 
# /  |  /  | /      \ /  |      /  \     /  | /      \ /  \  /  |
# $$ | /$$/ /$$$$$$  |$$ |      $$  \   /$$ |/$$$$$$  |$$  \ $$ |
# $$ |/$$/  $$ |__$$ |$$ |      $$$  \ /$$$ |$$ |__$$ |$$$  \$$ |
# $$  $$<   $$    $$ |$$ |      $$$$  /$$$$ |$$    $$ |$$$$  $$ |
# $$$$$  \  $$$$$$$$ |$$ |      $$ $$ $$/$$ |$$$$$$$$ |$$ $$ $$ |
# $$ |$$  \ $$ |  $$ |$$ |_____ $$ |$$$/ $$ |$$ |  $$ |$$ |$$$$ |
# $$ | $$  |$$ |  $$ |$$       |$$ | $/  $$ |$$ |  $$ |$$ | $$$ |
# $$/   $$/ $$/   $$/ $$$$$$$$/ $$/      $$/ $$/   $$/ $$/   $$/ 
                                                               
###############################################################################################################################################################
                                                               
def kalman_filter(z, Q=1e-5, R=0.01):
    """
    z - wejsciowy sygnal (obserwacje)
    Q - szum procesu
    R - szum pomiaru
    """
    n_iter = len(z)
    sz = (n_iter,)

    # Inicjalizacja zmiennych
    xhat = np.zeros(sz)      # Estymata a posteriori
    P = np.zeros(sz)         # Blad estymacji a posteriori
    xhatminus = np.zeros(sz) # Estymata a priori
    Pminus = np.zeros(sz)    # Blad estymacji a priori
    K = np.zeros(sz)         # Wzmocnienie Kalmana

    xhat[0] = z[0]
    P[0] = 1.0

    for k in range(1, n_iter):
        # Predykcja
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + Q

        # Aktualizacja
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]

    return xhat                                                       

def open_KALMAN_filter_popup_callback():
    dpg.show_item("kalman_filter_popup")

def apply_KALMAN_filter_callback():
    global audio_file, audio_file_filtered, audio_file_path_filtered, sampling_rate_filtered

    if audio_file is None:
        print("Najpierw wczytaj plik!")
        return

    prepare_environment_for_filtering()

    Q = float(dpg.get_value("kalman_Q"))
    R = float(dpg.get_value("kalman_R"))
    sampling_rate_filtered = sampling_rate

    start_benchmark()
    start_processing_indicator()

    if audio_file.ndim == 2:
        for channel in range(audio_file.shape[1]):
            audio_file_filtered[:, channel] = kalman_filter(audio_file[:, channel], Q, R)
    else:
        audio_file_filtered = kalman_filter(audio_file, Q, R)

    stop_benchmark_and_show_results()
    stop_processing_indicator()

    file_original_path = Path(audio_file_path)
    audio_file_path_filtered = os.path.join(os.path.dirname(audio_file_path), file_original_path.with_stem(f"{Path(audio_file_path).stem}_KALMAN_{Q}_{R}")) 

    show_filtered_plot_callback()
    show_filtered_spectrum_callback()
    call_create_audio_filtered_play_callback()

    print(f"Zastosowano filtracje Kalmana: Q={Q}, R={R}")
    print(f"Zapisywanie do pliku: {audio_file_path_filtered}")
    save_audio_with_convert(audio_file_path_filtered, sampling_rate, audio_file_filtered)
    print(f"Zapisano do pliku: {audio_file_path_filtered}")
    dpg.hide_item("kalman_filter_popup")

# Okno Popup Filtracji
with dpg.window(label="Filtracja Kalmana", tag="kalman_filter_popup", modal=True, show=False, width=800, height=400):
    dpg.add_input_float(label="Szum procesu (Q)", tag="kalman_Q", default_value=1e-5, min_value=1e-8, step=1e-5, max_value=1.0, format="%.8f")
    dpg.add_input_float(label="Szum pomiaru (R)", tag="kalman_R", default_value=0.01, min_value=1e-8, step=0.01, max_value=1.0, format="%.8f")
    dpg.add_button(label="Zastosuj filtr Kalmana", callback=apply_KALMAN_filter_callback)


###############################################################################################################################################################

#   ______   __    __  _______  
#  /      \ /  \  /  |/       \ 
# /$$$$$$  |$$  \ $$ |$$$$$$$  |
# $$ \__$$/ $$$  \$$ |$$ |__$$ |
# $$      \ $$$$  $$ |$$    $$< 
#  $$$$$$  |$$ $$ $$ |$$$$$$$  |
# /  \__$$ |$$ |$$$$ |$$ |  $$ |
# $$    $$/ $$ | $$$ |$$ |  $$ |
#  $$$$$$/  $$/   $$/ $$/   $$/ 
                                                                                                      
###############################################################################################################################################################
import librosa
from typing import Union, Tuple

def load_audio(
    path_or_array: Union[str, np.ndarray],
    sr: int = None
) -> Tuple[np.ndarray, int]:
    """
    Wczytuje plik WAV lub zwraca podana tablice.
    Jesli podano sciezke -> zwraca (y, sr), gdzie y.shape = (channels, samples).
    Jesli podano ndarray:
      -      jesli ksztalt (n,) -> mono
      -      jesli ksztalt (n,2) -> stereo
    """
    if isinstance(path_or_array, str):
        y, sr = librosa.load(path_or_array, sr=sr, mono=False)  # shape (channels, samples) lub (samples,)
        if y.ndim == 1:
            y = y[np.newaxis, :]  # (1, samples)
        return y, sr
    elif isinstance(path_or_array, np.ndarray):
        arr = path_or_array
        if arr.ndim == 1:
            return arr[np.newaxis, :], sr or 0
        elif arr.ndim == 2:
            # zakladamy (channels, samples)
            return arr, sr or 0
        else:
            raise ValueError("Tablica musi miec wymiar 1D (mono) lub 2D (stereo).")
    else:
        raise TypeError("path_or_array musi byc sciezka (str) lub ndarray.")

def calculate_snr2(
    original: Union[str, np.ndarray],
    noisy: Union[str, np.ndarray],
    sr: int = None,
    mode: str = 'mean'
) -> float:
    """
    Oblicza SNR (dB) miedzy sygnalem oryginalnym a zaszumionym/po filtracji.

    Parametry:
    - original: sciezka do WAV lub ndarray (mono lub stereo)
    - noisy:    sciezka do WAV lub ndarray (mono lub stereo)
    - sr:       zadana czestotliwosc probkowania (tylko przy ndarray=None)
    - mode:     jak zredukowac stereo do jednej wartosci:
                'mean'  - srednia SNR z kanalow
                'mono'  - najpierw miksuje do mono (srednia kanalow), potem liczy
                'perch' - zwraca liste wartosci [snr_ch0, snr_ch1, ...]

    Zwraca:
    - float (SNR w dB) albo liste floatow jesli mode='perch'
    """
    # Wczytanie
    sig_orig, sr1 = load_audio(original, sr)
    sig_noisy, sr2 = load_audio(noisy,    sr)
    if sr1 and sr2 and sr1 != sr2:
        raise ValueError(f"Rozne sr: {sr1} vs {sr2}")
    
    # Przytnij do tej samej dlugosci
    n = min(sig_orig.shape[1], sig_noisy.shape[1])
    sig_orig = sig_orig[:, :n]
    sig_noisy = sig_noisy[:, :n]

    # Oblicz szum
    noise = sig_noisy - sig_orig

    # Moc sygnalu i mocy szumu per kanal
    p_signal = np.mean(sig_orig**2, axis=1)
    p_noise  = np.mean(noise**2,    axis=1)

    # Unikamy dzielenia przez zero
    p_noise = np.where(p_noise == 0, np.finfo(float).eps, p_noise)

    # SNR per kanal (linia)
    snr_vals = 10 * np.log10(p_signal / p_noise)

    # Zwracanie zgodnie z trybem
    if mode == 'perch':
        return snr_vals.tolist()
    elif mode == 'mean':
        return float(np.mean(snr_vals))
    elif mode == 'mono':
        # miks do mono i liczenie na jednym kanale
        orig_mono = np.mean(sig_orig, axis=0)
        noisy_mono = np.mean(sig_noisy, axis=0)
        noise_mono = noisy_mono - orig_mono
        p_s = np.mean(orig_mono**2)
        p_n = np.mean(noise_mono**2)
        p_n = p_n if p_n != 0 else np.finfo(float).eps
        return float(10 * np.log10(p_s / p_n))
    else:
        raise ValueError("Nieznany tryb. Wybierz 'mean', 'mono' lub 'perch'.")

# — przyklad uzycia —

# # 1) Z plikow WAV:
# snr_db = calculate_snr("original.wav", "noisy.wav", mode='mean')
# print(f"SNR (srednie): {snr_db:.2f} dB")

# 2) Z tablic NumPy:
#    audio_orig = np.array([...])
#    audio_noisy = np.array([...])
#    snr_db = calculate_snr(audio_orig, audio_noisy, sr=44100, mode='perch')
#    print("SNR per channel:", snr_db)


# Funkcja obliczajaca kalsyczny zwykly snr
def calculate_snr(original, noisy_or_filtered):
    noise = noisy_or_filtered - original

    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float('inf')  # Idealny przypadek
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# Funkcja obliczajaca segmentowy snr
def compute_segmental_snr(original_signal, noisy_or_filtered, sampling_rate, frame_duration_ms=20):
    original_signal = original_signal.astype(np.float32)
    noisy_or_filtered = noisy_or_filtered.astype(np.float32)

    # Jesli dane sa w zakresie int16, znormalizuj:
    if np.max(np.abs(original_signal)) > 1.0:
        original_signal /= 32768.0
    if np.max(np.abs(noisy_or_filtered)) > 1.0:
        noisy_or_filtered /= 32768.0

    if original_signal.shape != noisy_or_filtered.shape:
        raise ValueError("Sygnaly musza miec ta sama dlugosc")

    frame_length = int(sampling_rate * frame_duration_ms / 1000)
    num_frames = len(original_signal) // frame_length

    segmental_snrs = []

    for i in range(num_frames):
        start = i * frame_length
        end = start + frame_length

        s = original_signal[start:end]
        n = noisy_or_filtered[start:end] - s

        signal_power = np.mean(s ** 2) + 1e-8  # Avoid div by 0
        noise_power = np.mean(n ** 2) + 1e-8

        snr = 10 * np.log10(signal_power / noise_power)
        segmental_snrs.append(snr)

    return np.mean(segmental_snrs)

# Funkcja obliczajaca SNR wazony krytycznie (Critical-band SNR) – uwzglednia, w jakim pasmie sluch jest wrazliwszy.
def compute_critical_band_snr(original_signal, noisy_or_filtered, sampling_rate):
    original_signal = original_signal.astype(np.float64)
    noisy_or_filtered = noisy_or_filtered.astype(np.float64)

    # print("original_signal:")
    # print(original_signal)
    # print(max(original_signal))

    # print("noisy_or_filtered:")
    # print(noisy_or_filtered)
    # print(max(noisy_or_filtered))

    if np.max(np.abs(original_signal)) > 1.0:
        original_signal /= 32768.0
        
    if np.max(np.abs(noisy_or_filtered)) > 1.0:
        noisy_or_filtered /= 32768.0

    if original_signal.shape != noisy_or_filtered.shape:
        raise ValueError("Signals must be the same length")

    bands = bark_band_filters(sampling_rate)
    cb_snr_values = []

    # print("original_signal:")
    # print(original_signal)
    # print("max: " + str(max(original_signal)))
    # print("min: " + str(min(original_signal)))

    # print("noisy_or_filtered:")
    # print(noisy_or_filtered)
    # print("max: " + str(max(noisy_or_filtered)))
    # print("min: " + str(min(noisy_or_filtered)))

    start_processing_indicator()
    try:
        for low, high in bands:
            orig_band = SNR_bandpass_filter(original_signal, low, high, sampling_rate)
            test_band = SNR_bandpass_filter(noisy_or_filtered, low, high, sampling_rate)

            noise_band = test_band - orig_band

            # print("========================================================================")
            # print("orig_band:")
            # print(orig_band)
            # print("max: " + str(max(orig_band)))
            # print("min: " + str(min(orig_band)))

            # print("========================================================================")
            # print("test_band:")
            # print(test_band)
            # print("max: " + str(max(test_band)))
            # print("min: " + str(min(test_band)))

            # print("========================================================================")
            # print("noise_band:")
            # print(noise_band)
            # print("max: " + str(max(noise_band)))
            # print("min: " + str(min(noise_band)))

            signal_power = np.mean(orig_band ** 2) + 1e-8
            noise_power = np.mean(noise_band ** 2) + 1e-8

            snr_band = 10 * np.log10(signal_power / noise_power)
            cb_snr_values.append(snr_band)
    except Exception as e:
        print(f"Blad podczas oblicznia compute_critical_band_snr: {e}")
        stop_processing_indicator()
        return 0

    stop_processing_indicator()
    return np.mean(cb_snr_values)

def bark_band_filters(sampling_rate):
    # Pasma w Hz – uproszczone bark bands
    bands = [
        (20, 100), (100, 200), (200, 300), (300, 400), (400, 510),
        (510, 630), (630, 770), (770, 920), (920, 1080), (1080, 1270),
        (1270, 1480), (1480, 1720), (1720, 2000), (2000, 2320),
        (2320, 2700), (2700, 3150), (3150, 3700), (3700, 4400),
        (4400, 5300), (5300, 6400), (6400, 7700), (7700, 9500),
        (9500, 12000), (12000, 15500)
    ]
    return bands

def SNR_bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    if low <= 0:
        low = 1e-5  # avoid log(0) or unstable filter
    if high >= 1:
        high = 0.9999  # stay within bounds

    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfilt(sos, signal)

# Funkcja wywolujaca obliczanie SNR i wyswetlajaca wynik
def show_snr_analysis():
    global audio_file_original, audio_file, sampling_rate

    if audio_file_original is None:
        print("Nie wczytano oryginalnego (niezaszumionego) sygnalu!")
        return

    if audio_file is None or audio_file_filtered is None:
        print("Brak danych: zaladuj plik wejsciowy i przefiltruj go.")
        return

    min_len = min(len(audio_file_original), len(audio_file), len(audio_file_filtered))

    snr_before = calculate_snr(audio_file_original[:min_len], audio_file[:min_len])
    snr_after = calculate_snr(audio_file_original[:min_len], audio_file_filtered[:min_len])

    snr_bf = calculate_snr2(audio_file_original[:min_len], audio_file[:min_len], sr=sampling_rate, mode='mean')
    snr_af = calculate_snr2(audio_file_original[:min_len], audio_file_filtered[:min_len], sr=sampling_rate, mode='mean')

    snr_segmental_before = compute_segmental_snr(audio_file_original[:min_len], audio_file[:min_len], sampling_rate)
    snr_segmental_after = compute_segmental_snr(audio_file_original[:min_len], audio_file_filtered[:min_len], sampling_rate)
    
    snr_cb_before = compute_critical_band_snr(audio_file_original[:min_len], audio_file[:min_len], sampling_rate)
    snr_cb_after = compute_critical_band_snr(audio_file_original[:min_len], audio_file_filtered[:min_len], sampling_rate)

    print(f"SNR przed filtracja: {snr_before:.2f} dB")
    print(f"SNR po filtracji:   {snr_after:.2f} dB")
    
    print(f"SNR2 przed filtracja: {snr_bf:.2f} dB")
    print(f"SNR2 po filtracji:   {snr_af:.2f} dB")

    print(f"SNR segmental przed filtracja: {snr_segmental_before:.2f} dB")
    print(f"SNR segmental po filtracji:   {snr_segmental_after:.2f} dB")

    print(f"SNR CB przed filtracja: {snr_cb_before:.2f} dB")
    print(f"SNR CB po filtracji:   {snr_cb_after:.2f} dB")

    with dpg.window(label="SNR Analiza", tag="SNRResultWindow", show=False, width=400, height=150):
        apply_window_geometry("SNRResultWindow", default_pos=(0, 672), default_size=(400, 150))
        dpg.add_text(f"SNR przed filtracja: {snr_before:.2f} dB\nSNR po filtracji: {snr_after:.2f} dB\n\n" +
                     f"SNR2 przed filtracja: {snr_bf:.2f} dB\nSNR2 po filtracji: {snr_af:.2f} dB\n\n" +
                     f"SNR segmental przed filtracja: {snr_segmental_before:.2f} dB\nSNR segmental po filtracji: {snr_segmental_after:.2f} dB\n\n" +
                     f"SNR CB przed filtracja: {snr_cb_before:.2f} dB\nSNR CB po filtracji: {snr_cb_after:.2f} dB")
    

###############################################################################################################################################################

#  ________   ______   _______   ______   ______          ______   __    __  _______   ______   ______  
# /        | /      \ /       \ /      | /      \        /      \ /  |  /  |/       \ /      | /      \ 
# $$$$$$$$/ /$$$$$$  |$$$$$$$  |$$$$$$/ /$$$$$$  |      /$$$$$$  |$$ |  $$ |$$$$$$$  |$$$$$$/ /$$$$$$  |
#     /$$/  $$ |__$$ |$$ |__$$ |  $$ |  $$ \__$$/       $$ |__$$ |$$ |  $$ |$$ |  $$ |  $$ |  $$ |  $$ |
#    /$$/   $$    $$ |$$    $$/   $$ |  $$      \       $$    $$ |$$ |  $$ |$$ |  $$ |  $$ |  $$ |  $$ |
#   /$$/    $$$$$$$$ |$$$$$$$/    $$ |   $$$$$$  |      $$$$$$$$ |$$ |  $$ |$$ |  $$ |  $$ |  $$ |  $$ |
#  /$$/____ $$ |  $$ |$$ |       _$$ |_ /  \__$$ |      $$ |  $$ |$$ \__$$ |$$ |__$$ | _$$ |_ $$ \__$$ |
# /$$      |$$ |  $$ |$$ |      / $$   |$$    $$/       $$ |  $$ |$$    $$/ $$    $$/ / $$   |$$    $$/ 
# $$$$$$$$/ $$/   $$/ $$/       $$$$$$/  $$$$$$/        $$/   $$/  $$$$$$/  $$$$$$$/  $$$$$$/  $$$$$$/  
                                                                                                      
###############################################################################################################################################################

# Zapis przefiltrowanego sygnalu do pliku
def save_audio(file_path, sampling_freq, data):
    write(file_path, sampling_freq, data.astype(np.int16))

def save_audio_with_convert(file_path, sampling_freq, data):

    if isinstance(sampling_freq, float):
        sampling_freq = int(sampling_freq)  # Konwersja na int

    if data.dtype != np.int16:
        data = np.clip(data, -1.0, 1.0)  # Upewniamy sie ze wartosci sa w zakresie
        data = np.int16(data * 32767)    # Skalowanie do 16 bit

    write(file_path, sampling_freq, data)
    print(f"Zapisano plik: {file_path} przy {sampling_freq}Hz")


###############################################################################################################################################################

#   ______   __    __  _______   ______   ______         _______   __         ______   __      __  ________  _______  
#  /      \ /  |  /  |/       \ /      | /      \       /       \ /  |       /      \ /  \    /  |/        |/       \ 
# /$$$$$$  |$$ |  $$ |$$$$$$$  |$$$$$$/ /$$$$$$  |      $$$$$$$  |$$ |      /$$$$$$  |$$  \  /$$/ $$$$$$$$/ $$$$$$$  |
# $$ |__$$ |$$ |  $$ |$$ |  $$ |  $$ |  $$ |  $$ |      $$ |__$$ |$$ |      $$ |__$$ | $$  \/$$/  $$ |__    $$ |__$$ |
# $$    $$ |$$ |  $$ |$$ |  $$ |  $$ |  $$ |  $$ |      $$    $$/ $$ |      $$    $$ |  $$  $$/   $$    |   $$    $$< 
# $$$$$$$$ |$$ |  $$ |$$ |  $$ |  $$ |  $$ |  $$ |      $$$$$$$/  $$ |      $$$$$$$$ |   $$$$/    $$$$$/    $$$$$$$  |
# $$ |  $$ |$$ \__$$ |$$ |__$$ | _$$ |_ $$ \__$$ |      $$ |      $$ |_____ $$ |  $$ |    $$ |    $$ |_____ $$ |  $$ |
# $$ |  $$ |$$    $$/ $$    $$/ / $$   |$$    $$/       $$ |      $$       |$$ |  $$ |    $$ |    $$       |$$ |  $$ |
# $$/   $$/  $$$$$$/  $$$$$$$/  $$$$$$/  $$$$$$/        $$/       $$$$$$$$/ $$/   $$/     $$/     $$$$$$$$/ $$/   $$/ 
                                                    
###############################################################################################################################################################

# Globalne zmienne
playback_stream = None
is_playing = {
    "input": False,
    "filtered": False
}

# Stan odtwarzania
current_positions = {
    "input": 0.0,
    "filtered": 0.0
}

# Start odtwarzania
def start_playback(audio_data, sampling_rate, label, window_tag):
    global playback_stream, current_positions, is_playing

    is_filtered = np.array_equal(audio_data, audio_file_filtered)
    signal_key = "filtered" if is_filtered else "input"

    if is_playing[signal_key]:
        print(f"Currently playing {signal_key}")
        return

    current_positions[signal_key] = 0.0
    is_playing[signal_key] = True

    def callback(outdata, frames, time, status):
        nonlocal is_filtered, signal_key

        if status:
            print(status)

        start_idx = int(current_positions[signal_key] * sampling_rate)
        end_idx = start_idx + frames

        chunk = audio_data[start_idx:end_idx]

        if len(chunk) < frames:
            outdata[:len(chunk)] = chunk[:, None]
            outdata[len(chunk):] = 0
            is_playing[signal_key] = False
            playback_stream.stop()
            return
        else:
            outdata[:] = chunk[:, None]

        current_positions[signal_key] += frames / sampling_rate

    # Uruchom monitoring progresu
    threading.Thread(
        target=monitor_progress,
        args=(window_tag, audio_data, lambda: current_positions[signal_key]),
        daemon=True
    ).start()

    playback_stream = sd.OutputStream(
            callback=callback,
            channels=1,
            samplerate=sampling_rate,
            blocksize=1024,  # Optional: control how much data is read per callback
            latency='low'    # Try 'high' if underflows continue
        )    
    playback_stream.start()
    
# Stop odtwarzania
def stop_playback(audio_data):
    global playback_stream, is_playing
    if playback_stream:
        playback_stream.stop()
        playback_stream.close()
        playback_stream = None

    if np.array_equal(audio_data, audio_file_filtered):
        is_playing["filtered"] = False
    else:
        is_playing["input"] = False

    print("Odtwarzanie zatrzymane!")

# Funkcja seek
def seek_audio(sender, app_data, user_data):
    signal_type, slider_tag = user_data
    new_position = dpg.get_value(slider_tag)
    current_positions[signal_type] = new_position
    # print(f"Seek {signal_type}: {new_position:.2f} s")

# Aktualizacja progress bara
def monitor_progress(window_tag, audio_data, get_current_position):
    global is_playing

    play = None
    if np.array_equal(audio_data, audio_file_filtered):
        play = "filtered"
    else:
        play = "input"

    while is_playing[play] and dpg.does_item_exist(window_tag):
        try:
            # Bezpieczny odczyt pozycji
            current_pos = get_current_position()

            # Aktualizacja progres bara
            progress = current_pos / (len(audio_data) / effective_sampling_rate(audio_data))
            dpg.set_value(f"{window_tag}_progress_bar", progress)
            dpg.set_value(f"{window_tag}_progess_slider_value", current_pos)

        except Exception as e:
            print(f"Blad w monitor_progress: {e}")
            break

        time.sleep(0.1)  # Aktualizuj co 100 ms

def effective_sampling_rate(audio_data):
    if np.array_equal(audio_data, audio_file_filtered):
        return sampling_rate_filtered if sampling_rate_filtered is not None else sampling_rate
    else:
        return sampling_rate

# GUI - Tworzenie okna
def create_audio_controls_window(windowTag, label, audio_data, sampling_rate, signal_type):
    if audio_data is None or audio_data.size == 0:
        print("Blad: Brak danych audio!")
        return

    if dpg.does_item_exist(windowTag):
        dpg.delete_item(windowTag)

    with dpg.window(tag=windowTag, label=label):
        apply_window_geometry(windowTag, default_pos=(820, 270), default_size=(500, 250))

        with dpg.group():
            with dpg.group(horizontal=True):
                dpg.add_button(label="Start", callback=lambda: start_playback(audio_data, sampling_rate, label, windowTag))
                dpg.add_button(label="Stop", width=120, callback=lambda: stop_playback(audio_data))

            dpg.add_progress_bar(label="Postep", tag=f"{windowTag}_progress_bar", width=480)
            dpg.add_slider_float(
                label="", 
                tag=f"{windowTag}_progess_slider_value",
                width=480, 
                min_value=0.0, 
                max_value=len(audio_data) / sampling_rate, 
                default_value=0.0, 
                callback=seek_audio,
                user_data=(signal_type, f"{windowTag}_progess_slider_value")
            )


###############################################################################################################################################################

#   ______   ______  __    __         ______   ________  __    __  ________  _______    ______   ________  ______   _______  
#  /      \ /      |/  \  /  |       /      \ /        |/  \  /  |/        |/       \  /      \ /        |/      \ /       \ 
# /$$$$$$  |$$$$$$/ $$  \ $$ |      /$$$$$$  |$$$$$$$$/ $$  \ $$ |$$$$$$$$/ $$$$$$$  |/$$$$$$  |$$$$$$$$//$$$$$$  |$$$$$$$  |
# $$ \__$$/   $$ |  $$$  \$$ |      $$ | _$$/ $$ |__    $$$  \$$ |$$ |__    $$ |__$$ |$$ |__$$ |   $$ |  $$ |  $$ |$$ |__$$ |
# $$      \   $$ |  $$$$  $$ |      $$ |/    |$$    |   $$$$  $$ |$$    |   $$    $$< $$    $$ |   $$ |  $$ |  $$ |$$    $$< 
#  $$$$$$  |  $$ |  $$ $$ $$ |      $$ |$$$$ |$$$$$/    $$ $$ $$ |$$$$$/    $$$$$$$  |$$$$$$$$ |   $$ |  $$ |  $$ |$$$$$$$  |
# /  \__$$ | _$$ |_ $$ |$$$$ |      $$ \__$$ |$$ |_____ $$ |$$$$ |$$ |_____ $$ |  $$ |$$ |  $$ |   $$ |  $$ \__$$ |$$ |  $$ |
# $$    $$/ / $$   |$$ | $$$ |      $$    $$/ $$       |$$ | $$$ |$$       |$$ |  $$ |$$ |  $$ |   $$ |  $$    $$/ $$ |  $$ |
#  $$$$$$/  $$$$$$/ $$/   $$/        $$$$$$/  $$$$$$$$/ $$/   $$/ $$$$$$$$/ $$/   $$/ $$/   $$/    $$/    $$$$$$/  $$/   $$/ 
                                                                                                                           
###############################################################################################################################################################
import soundfile as sf

def generate_sine_wave(frequencies, duration, amplitude=0.5, sampling_rate=44100):
    """
    frequencies: float lub lista floatow [Hz]
    duration: czas w sekundach
    amplitude: od 0 do 1
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    if isinstance(frequencies, (float, int)):
        signal = amplitude * np.sin(2 * np.pi * frequencies * t)
    else:
        signal = np.zeros_like(t)
        for f in frequencies:
            signal += amplitude * np.sin(2 * np.pi * f * t)
        signal /= len(frequencies)  # normalizacja
    return signal.astype(np.float32)

def save_generated_signal(signal, sampling_rate, filename):
    sf.write(filename, signal, sampling_rate)
    print(f"Zapisano plik: {filename}")

def generate_and_save_callback():
    freq_str = dpg.get_value("gen_freq")
    duration = dpg.get_value("gen_duration")
    amplitude = dpg.get_value("gen_amp")
    sr = dpg.get_value("gen_sr")
    filename = dpg.get_value("gen_filename")

    try:
        frequencies = [float(f.strip()) for f in freq_str.split(",")]
    except:
        print("Blad: podano nieprawidlowe czestotliwosci")
        return

    signal = generate_sine_wave(frequencies, duration, amplitude, sr)
    save_generated_signal(signal, sr, filename)

def open_gen_signal_window():
    if dpg.does_item_exist("SinWaveGeneratorWindow"):
        dpg.delete_item("SinWaveGeneratorWindow")

    with dpg.window(label="Generator sygnalu", tag="SinWaveGeneratorWindow"):
        apply_window_geometry("SinWaveGeneratorWindow", default_pos=(820, 270), default_size=(500, 250))
        dpg.add_input_text(label="Czestotliwosci [Hz] (np. 440 lub 440,880)", tag="gen_freq")
        dpg.add_input_float(label="Czas trwania [s]", default_value=2.0, tag="gen_duration")
        dpg.add_input_float(label="Amplituda", default_value=0.5, min_value=0.0, max_value=1.0, tag="gen_amp")
        dpg.add_input_int(label="Czestotliwosc probkowania", default_value=44100, tag="gen_sr")
        dpg.add_input_text(label="Nazwa pliku", default_value="generated.wav", tag="gen_filename")
        dpg.add_button(label="Generuj i zapisz", callback=generate_and_save_callback)


###############################################################################################################################################################

#  ________   ______    ______   ________  __    __  __       __  ______   ______   __    __  ______  ________ 
# /        | /      \  /      \ /        |/  |  /  |/  \     /  |/      | /      \ /  \  /  |/      |/        |
# $$$$$$$$/ /$$$$$$  |/$$$$$$  |$$$$$$$$/ $$ |  $$ |$$  \   /$$ |$$$$$$/ /$$$$$$  |$$  \ $$ |$$$$$$/ $$$$$$$$/ 
#     /$$/  $$ |__$$ |$$ \__$$/     /$$/  $$ |  $$ |$$$  \ /$$$ |  $$ |  $$ |__$$ |$$$  \$$ |  $$ |  $$ |__    
#    /$$/   $$    $$ |$$      \    /$$/   $$ |  $$ |$$$$  /$$$$ |  $$ |  $$    $$ |$$$$  $$ |  $$ |  $$    |   
#   /$$/    $$$$$$$$ | $$$$$$  |  /$$/    $$ |  $$ |$$ $$ $$/$$ |  $$ |  $$$$$$$$ |$$ $$ $$ |  $$ |  $$$$$/    
#  /$$/____ $$ |  $$ |/  \__$$ | /$$/____ $$ \__$$ |$$ |$$$/ $$ | _$$ |_ $$ |  $$ |$$ |$$$$ | _$$ |_ $$ |_____ 
# /$$      |$$ |  $$ |$$    $$/ /$$      |$$    $$/ $$ | $/  $$ |/ $$   |$$ |  $$ |$$ | $$$ |/ $$   |$$       |
# $$$$$$$$/ $$/   $$/  $$$$$$/  $$$$$$$$/  $$$$$$/  $$/      $$/ $$$$$$/ $$/   $$/ $$/   $$/ $$$$$$/ $$$$$$$$/ 
                                                                                                             
###############################################################################################################################################################
from scipy.signal import lfilter
audio_file_noisy = None

def add_noise(signal, noise_type="white", snr_db=10, sampling_rate=44100):
    signal = np.array(signal, dtype=np.float32)
    power_signal = np.mean(signal**2)
    
    if noise_type == "white":
        noise = np.random.normal(0, 1, len(signal))
    
    elif noise_type == "pink": # Voss-McCartney pink noise generation (simplified)
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        white = np.random.randn(len(signal))
        noise = lfilter(b, a, white)

    elif noise_type == "urban":
        low_freq = np.sin(2 * np.pi * 50 * np.linspace(0, len(signal)/sampling_rate, len(signal)))
        noise = 0.3 * low_freq + 0.7 * np.random.normal(0, 1, len(signal))

    elif noise_type == "industrial":
        tone = np.sin(2 * np.pi * 200 * np.linspace(0, len(signal)/sampling_rate, len(signal)))
        mod = np.random.normal(0, 1, len(signal)) * np.sin(2 * np.pi * 2 * np.linspace(0, len(signal)/sampling_rate, len(signal)))
        noise = tone * 0.4 + mod * 0.6

    elif noise_type == "impulse":
        noise = np.zeros_like(signal)
        impulse_count = len(signal) // 1000
        positions = np.random.choice(len(signal), impulse_count, replace=False)
        noise[positions] = np.random.uniform(-1, 1, size=impulse_count)

    else:
        raise ValueError(f"Nieobslugiwany typ szumu: {noise_type}")

    # Skalowanie do zadanego SNR
    power_noise = np.mean(noise**2)
    target_noise_power = power_signal / (10 ** (snr_db / 10))
    scaling_factor = np.sqrt(target_noise_power / (power_noise + 1e-8))
    noise_scaled = noise * scaling_factor

    return signal + noise_scaled

def add_noise_callback():
    global audio_file, audio_file_noisy

    if audio_file is None:
        print("Najpierw wczytaj sygnal!")
        return

    sampling_ratee = 44100
    noise_type = dpg.get_value("noise_type")
    snr_value = float(dpg.get_value("snr_value"))
    noised_filename = dpg.get_value("noised_filename")

    audio_file_noisy = add_noise(audio_file, noise_type=noise_type, snr_db=snr_value, sampling_rate=sampling_ratee)

    filename = os.path.join(os.path.dirname(noised_filename), f"{Path(noised_filename).stem}_{noise_type.upper()}_{str(snr_value)}.wav")

    print(f"Szum typu '{noise_type}' dodany z SNR = {snr_value} dB.")
    save_audio_with_convert(filename, sampling_ratee, audio_file_noisy)

def open_add_noise_window_callback():
    if dpg.does_item_exist("AddNoiseWindow"):
        dpg.delete_item("AddNoiseWindow")

    # print(Path(audio_file_path))
    # print(Path(audio_file_path).stem)
    # print(os.path.dirname(audio_file_path))

    filename = os.path.join(os.path.dirname(audio_file_path), f"{Path(audio_file_path).stem}_noised.wav")

    with dpg.window(label="Dodaj szum", tag="AddNoiseWindow"):
        apply_window_geometry("AddNoiseWindow", default_pos=(0, 0), default_size=(500, 250))
        dpg.add_combo(label="Rodzaj szumu", items=["white", "pink", "urban", "industrial", "impulse"], default_value="white", tag="noise_type")
        dpg.add_input_double(label="SNR [dB]", default_value=10.0, min_value=-10.0, max_value=40.0, step=0.1, tag="snr_value")
        dpg.add_input_text(label="Nazwa pliku", default_value=filename, tag="noised_filename")
        dpg.add_button(label="Wybierz plik", callback=lambda: open_save_file_dialog(filename))
        dpg.add_button(label="Zaszum sygnal", callback=add_noise_callback)


###############################################################################################################################################################

#  __       __   ______   ______  __    __ 
# /  \     /  | /      \ /      |/  \  /  |
# $$  \   /$$ |/$$$$$$  |$$$$$$/ $$  \ $$ |
# $$$  \ /$$$ |$$ |__$$ |  $$ |  $$$  \$$ |
# $$$$  /$$$$ |$$    $$ |  $$ |  $$$$  $$ |
# $$ $$ $$/$$ |$$$$$$$$ |  $$ |  $$ $$ $$ |
# $$ |$$$/ $$ |$$ |  $$ | _$$ |_ $$ |$$$$ |
# $$ | $/  $$ |$$ |  $$ |/ $$   |$$ | $$$ |
# $$/      $$/ $$/   $$/ $$$$$$/ $$/   $$/ 
                                                                                                      
###############################################################################################################################################################

# # Wczytaj plik z sygnalem i plik z szumem
# signal_waveform, sr = torchaudio.load(os.path.join(CURRENT_DIR, 'atlas_ATC_around_the_world.wav'))
# noise_waveform, sr_noise = torchaudio.load(os.path.join(CURRENT_DIR, 'atlas_ATC_around_the_world_noised_Pink.wav'))
# filtered_waveform, sr_filtered_noise = torchaudio.load(os.path.join(CURRENT_DIR, 'atlas_ATC_around_the_world_noised_Pink_BAND_Butterworth_4_300-5000.wav'))

# # Upewnij sie, ze oba sygnaly maja te sama dlugosc
# min_length = min(signal_waveform.shape[1], noise_waveform.shape[1],  filtered_waveform.shape[1])
# signal_waveform = signal_waveform[:, :min_length]
# noise_waveform = noise_waveform[:, :min_length]
# filtered_waveform = filtered_waveform[:, :min_length]

# filtered_waveform = filtered_waveform.repeat(2, 1)

# # Oblicz SNR
# snr_value = signal_noise_ratio(signal_waveform, noise_waveform)
# snr_filtered_value = signal_noise_ratio(signal_waveform, filtered_waveform)

# # srednia po kanalach
# snr_mean = snr_value.mean().item()
# print(f"srednie snr_value SNR (stereo): {snr_mean:.2f} dB")

# snr_mean_filtered = snr_filtered_value.mean().item()
# print(f"srednie snr_filtered_value SNR (stereo): {snr_mean_filtered:.2f} dB")

def delete_item(tag):
    if (dpg.does_item_exist(tag)):
        dpg.delete_item(tag)

def prepare_environment_for_filtering():
    global BeforeAfterPlotWindowCollapse, DifferenceAmplitudeWindowCollapse, BeforeAfterFrequencyDomainWindowCollapse, DifferenceFrequencyDomainWindowCollapse
    if dpg.does_item_exist("BeforeAfterPlotWindow"):
        BeforeAfterPlotWindowCollapse=dpg.get_item_state("BeforeAfterPlotWindow")["toggled_open"]
        dpg.delete_item("BeforeAfterPlotWindow")
    if dpg.does_item_exist("DifferenceAmplitudeWindow"):
        DifferenceAmplitudeWindowCollapse=dpg.get_item_state("DifferenceAmplitudeWindow")["toggled_open"]
        dpg.delete_item("DifferenceAmplitudeWindow")
    if dpg.does_item_exist("BeforeAfterFrequencyDomainWindow"):
        BeforeAfterFrequencyDomainWindowCollapse=dpg.get_item_state("BeforeAfterFrequencyDomainWindow")["toggled_open"]
        dpg.delete_item("BeforeAfterFrequencyDomainWindow")
    if dpg.does_item_exist("DifferenceFrequencyDomainWindow"):
        DifferenceFrequencyDomainWindowCollapse=dpg.get_item_state("DifferenceFrequencyDomainWindow")["toggled_open"]
        dpg.delete_item("DifferenceFrequencyDomainWindow")
    if dpg.does_item_exist("AmplitudeWindowFiltered"):
        dpg.delete_item("AmplitudeWindowFiltered")
    if dpg.does_item_exist("SpectrumWindowFiltered"):
        dpg.delete_item("SpectrumWindowFiltered")
    if dpg.does_item_exist("FilteredSignalPlaybackWindow"):
        dpg.delete_item("FilteredSignalPlaybackWindow")
    

tableHeight = 760
tableWidth = 760
viewport_last_size = [dpg.get_viewport_width(), dpg.get_viewport_height()]

tracked_tags = [
    "MainWindow",                       # okno menu
    "BenchmarkWindow",                  # okno statystyk
    "AmplitudeWindow",                  # wykres amplitudy wczytanego sygnału
    "AmplitudeWindowFiltered",          # wykres amplitudy przefiltrowaneo sygnału
    "SpectrumWindow",                   # wykres częstotliwości wczytanego sygnału
    "SpectrumWindowFiltered",           # wykres częstotliwości przefiltrowaneo sygnału
    "InputSignalPlaybackWindow",        # okno odtwarzania sygnału wczytanego
    "FilteredSignalPlaybackWindow",     # okno odtwarzania sygnału przefiltrowanego
    "BeforeAfterPlotWindow",            # wykres porownania amplitud sygnalow 
    "DifferenceAmplitudeWindow",        # wykres roznicy amplitud sygnalow z sygnalem oryginalnym
    "BeforeAfterFrequencyDomainWindow", # wykres porownania czestotliwosci sygnalow 
    "DifferenceFrequencyDomainWindow",  # wykres roznicy czestotilwosci sygnalow z sygnalem oryginalnym
    "SNRResultWindow",                  # okno z obliczonymi wartosciami SNR
    "SinWaveGeneratorWindow",           # okno generatora fal sinusoidalnych
    "AddNoiseWindow",                   # okno zaszumiania sygnalu 
    "OriginalAmplitudeWindow",          # wykres amplitudy sygnalu niezaszumionego (oryginalu)
    "OriginalSpectrumWindow",           # wykres czestotliwosci sygnalu niezaszumionego (oryginalu)
]

load_window_config()

def save_window_config_callback():
    for tag in tracked_tags: 
        if dpg.does_item_exist(tag):
            print(f"Saving window: {tag}")
            store_geometry_on_close("save_layout", None, tag) 

    save_window_config()

def save_noised_file_dialog_callback(sender, app_data):
    # app_data zawiera wybrane pliki, np. {'file_path_name': ..., 'file_name': ..., 'current_path': ...}
    selected_path = app_data['file_path_name']  # pelna sciezka
    # Ustaw wartosc w input_text o tagu "noised_filename"
    dpg.set_value("noised_filename", selected_path)

def call_create_audio_play_callback():
    create_audio_controls_window("InputSignalPlaybackWindow", "Odtwarzanie wczytanego pliku", audio_file, sampling_rate, "input")

def call_create_audio_filtered_play_callback():
    create_audio_controls_window("FilteredSignalPlaybackWindow", "Odtwarzanie przefiltrowanego pliku", audio_file_filtered, sampling_rate_filtered if sampling_rate_filtered is not None else sampling_rate, "filtered")

def open_load_dialog():
    dpg.show_item("load_wav_file_dialog")

def open_load_original_dialog():
    dpg.show_item("load_original_wav_file_dialog")

def open_save_file_dialog(path):
    dpg.show_item("save_wav_file_dialog")

def check_resize():
    global viewport_last_size
    current_size = [dpg.get_viewport_width(), dpg.get_viewport_height()]
    if current_size != viewport_last_size:
        # print(f"Viewport resized: {current_size}")
        change_windows_size(current_size)
        viewport_last_size = current_size

def vieport_resized():
    global viewport_last_size
    try:
        check_resize()
        change_windows_size(viewport_last_size)
    except Exception as ex:
        print(ex)

def change_windows_size(viewport_size):
    global viewport_last_size
    if viewport_size is None:
        viewport_size = viewport_last_size
    try:
        menu_width = dpg.get_item_width("MainWindow")
        window_correction = -5

        if dpg.does_item_exist("AmplitudeWindow"):
            item_new_width=(viewport_size[0] - menu_width) / 2 + window_correction
            dpg.set_item_width("AmplitudeWindow", item_new_width)

        if dpg.does_item_exist("AmplitudeWindowFiltered"):
            item_new_width=(viewport_size[0] - menu_width) / 2 + 3 * window_correction
            dpg.set_item_width("AmplitudeWindowFiltered", item_new_width)
            curr_pos = dpg.get_item_pos("AmplitudeWindowFiltered")
            dpg.set_item_pos("AmplitudeWindowFiltered", [menu_width + item_new_width - 3 * window_correction, curr_pos[1]])
            if dpg.does_item_exist("OriginalAmplitudeWindow"):
                dpg.set_item_width("OriginalAmplitudeWindow", item_new_width)
                dpg.set_item_pos("OriginalAmplitudeWindow", curr_pos)

        if dpg.does_item_exist("SpectrumWindow"):
            item_new_width=(viewport_size[0] - menu_width) / 2 + window_correction
            dpg.set_item_width("SpectrumWindow", item_new_width)

        if dpg.does_item_exist("SpectrumWindowFiltered"):
            item_new_width=(viewport_size[0] - menu_width) / 2 + 3 * window_correction
            dpg.set_item_width("SpectrumWindowFiltered", item_new_width)
            curr_pos = dpg.get_item_pos("SpectrumWindowFiltered")
            dpg.set_item_pos("SpectrumWindowFiltered", [menu_width + item_new_width - 3 * window_correction, curr_pos[1]])
            if dpg.does_item_exist("OriginalSpectrumWindow"):
                dpg.set_item_width("OriginalSpectrumWindow", item_new_width)
                dpg.set_item_pos("OriginalSpectrumWindow", curr_pos)

        if dpg.does_item_exist("InputSignalPlaybackWindow"):
            item_new_width=(viewport_size[0] - menu_width) / 2 + window_correction
            dpg.set_item_width("InputSignalPlaybackWindow", item_new_width)

        if dpg.does_item_exist("FilteredSignalPlaybackWindow"):
            item_new_width=(viewport_size[0] - menu_width) / 2 + 3 * window_correction
            dpg.set_item_width("FilteredSignalPlaybackWindow", item_new_width)
            curr_pos = dpg.get_item_pos("FilteredSignalPlaybackWindow")
            dpg.set_item_pos("FilteredSignalPlaybackWindow", [menu_width + item_new_width - 3 * window_correction, curr_pos[1]])


        if dpg.does_item_exist("BeforeAfterPlotWindow"):
            item_new_width=viewport_size[0] - menu_width + 4 * window_correction
            dpg.set_item_width("BeforeAfterPlotWindow", item_new_width)

        if dpg.does_item_exist("DifferenceAmplitudeWindow"):
            item_new_width=viewport_size[0] - menu_width + 4 * window_correction
            dpg.set_item_width("DifferenceAmplitudeWindow", item_new_width)

        if dpg.does_item_exist("BeforeAfterFrequencyDomainWindow"):
            item_new_width=viewport_size[0] - menu_width + 4 * window_correction
            dpg.set_item_width("BeforeAfterFrequencyDomainWindow", item_new_width)

        if dpg.does_item_exist("DifferenceFrequencyDomainWindow"):
            item_new_width=viewport_size[0] - menu_width + 4 * window_correction
            dpg.set_item_width("DifferenceFrequencyDomainWindow", item_new_width)

    except Exception as ex:
        print(ex)

def clear_loaded_audio_data():
    global audio_file_original
    global sampling_rate_original
    global audio_file_path_original
    global audio_file
    global sampling_rate
    global audio_file_path
    global audio_file_filtered_save_path
    global audio_file_filtered
    global sampling_rate_filtered
    global audio_file_path_filtered

    audio_file_original = None
    sampling_rate_original = None
    audio_file_path_original = None
    audio_file = None
    sampling_rate = None
    audio_file_path = None
    audio_file_filtered_save_path = None
    audio_file_filtered = None
    sampling_rate_filtered = None
    audio_file_path_filtered = None

    prepare_environment_for_filtering()

with dpg.file_dialog(directory_selector=False,
                     show=False, 
                     modal=True, 
                     width=900, 
                     height=500, 
                     label="Wczytywanie pliku do odszumienia", 
                     callback=load_wav_file_callback, 
                     tag="load_wav_file_dialog"):
    dpg.add_file_extension(".wav", color=(150, 255, 150, 255))
    dpg.add_file_extension(".*")

with dpg.file_dialog(directory_selector=False, 
                     show=False, 
                     modal=True, 
                     width=900,
                     height=500, 
                     label="Wczytywanie oryginalnego (niezaszumionego) pliku", 
                     callback=load_original_wav_file_callback, 
                     tag="load_original_wav_file_dialog"):
    dpg.add_file_extension(".wav", color=(150, 255, 150, 255))
    dpg.add_file_extension(".*")

with dpg.file_dialog(directory_selector=False, 
                     show=False, 
                     modal=True, 
                     width=900,
                     height=500,
                     label="Zapisywanie pliku .wav", 
                     callback=save_noised_file_dialog_callback, 
                     tag="save_wav_file_dialog"):
    dpg.add_file_extension(".wav", color=(150, 255, 150, 255))
    dpg.add_file_extension(".*")

with dpg.window(label="Menu", tag="MainWindow", no_resize=False, no_close=True):    
    apply_window_geometry("MainWindow", default_pos=(0, 0), default_size=(tableWidth, tableHeight))
    with dpg.group(horizontal=True):
        with dpg.child_window(autosize_y=True):
            dpg.add_button(label="Wczytaj plik oryginal (.wav)", callback=open_load_original_dialog)
            dpg.add_button(label="Wczytaj plik do filtracji (.wav)", callback=open_load_dialog)

            with dpg.collapsing_header(label="Filtry", default_open=True):
                dpg.add_button(label="Filtracja dolnoprzepustowa", callback=open_LOWPASS_filter_popup_callback)
                dpg.add_button(label="Filtracja gornoprzepustowa", callback=open_HIPASS_filter_popup_callback)
                dpg.add_button(label="Filtracja pasmoprzepustowa", callback=open_BANDPASS_filter_popup_callback)
                dpg.add_button(label="Filtracja LMS", callback=open_LMS_filter_popup_callback)
                dpg.add_button(label="Filtracja falkowa", callback=open_WAVELET_filter_popup_callback)
                dpg.add_button(label="Filtracja kalmana", callback=open_KALMAN_filter_popup_callback)
                # dpg.add_button(label="Filtracja AI", callback=open_AI_filter_popup_callback)

            with dpg.collapsing_header(label="Wykresy", default_open=True):
                dpg.add_button(label="Pokaz roznice sygnalow", callback=show_signal_difference)

                dpg.add_button(label="Pokaz wykres amplitudy", callback=show_plot_callback)
                dpg.add_button(label="Pokaz wykres czestotliwosci", callback=show_spectrum_callback)

                dpg.add_button(label="Pokaz wykres amplitudy po filtracji", callback=show_filtered_plot_callback)
                dpg.add_button(label="Pokaz wykres czestotliwosci po filtracji", callback=show_filtered_spectrum_callback)

                dpg.add_button(label="Pokaz wykres amplitudy orginalnego pliku", callback=show_original_plot_callback)
                dpg.add_button(label="Pokaz wykres czestotliwosci orginalnego pliku", callback=show_original_spectrum_callback)

            with dpg.collapsing_header(label="SNR", default_open=True):
                dpg.add_button(label="Oblicz SNR", callback=show_snr_analysis)

            with dpg.collapsing_header(label="Audio", default_open=True):
                dpg.add_button(label="Audio playback", callback=call_create_audio_play_callback)
                dpg.add_button(label="Audio filter playback", callback=call_create_audio_filtered_play_callback)

            with dpg.collapsing_header(label="Ustawienia", default_open=False):
                dpg.add_button(label="Zapisz layout", tag="save_layout", callback=save_window_config_callback)
                dpg.add_button(label="Generuj sygnal sin", callback=open_gen_signal_window)
                dpg.add_button(label="Zaszum plik", callback=open_add_noise_window_callback)
                dpg.add_button(label="Wyczysc wczytane dane", callback=lambda: clear_loaded_audio_data())
                     

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.toggle_viewport_fullscreen()
dpg.set_viewport_resize_callback(vieport_resized)
dpg.maximize_viewport()

# Custom main loop
while dpg.is_dearpygui_running():
    dpg.render_dearpygui_frame()

# save_window_config()
dpg.destroy_context()