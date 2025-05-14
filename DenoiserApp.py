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
# import tensorflow as tf
import torch
import torchaudio
import librosa
matplotlib.use('Agg')
from scipy.io.wavfile import read, write
from scipy.signal import butter, bessel, cheby1, filtfilt, freqz, lfilter, sosfilt
from scipy.fft import fft, fftfreq
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

audio_file_original = None  # Oryginalny, niezaszumiony sygnał
sampling_rate_original = None  # Oryginalny, niezaszumiony sygnał - sampling rate

window_config = {}
audio_file = None  # Tutaj będzie zapisany wczytany plik audio, ktory bedzie podlegal odszumianiu
sampling_rate = None
audio_file_path = None 

audio_file_filtered = None  # Tutaj będzie zapisany nasz przefiltrowany plik audio
sampling_rate_filtered = None
audio_file_path_filtered = None 

# GUI Setup
dpg.create_context()
dpg.create_viewport(title='Denoiser App', width=1600, height=900)
dpg.toggle_viewport_fullscreen()
dpg.maximize_viewport()

def log_error(s):   # nice log xD
    print(s)


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

def start_benchmark():
    global filter_start_time, start_cpu_percent, start_memory_info
    filter_start_time = time.time()
    start_cpu_percent = psutil.cpu_percent(interval=None)
    start_memory_info = process.memory_info().rss  # RAM w bajtach używany przez nasz proces

def stop_benchmark_and_show_results():
    global filter_start_time, start_cpu_percent, start_memory_info
    if filter_start_time is None:
        print("Benchmark nie został rozpoczęty!")
        return

    # Czas trwania
    elapsed_time = time.time() - filter_start_time

    # Aktualne zużycie CPU i RAM
    end_cpu_percent = psutil.cpu_percent(interval=None)
    end_memory_info = process.memory_info().rss

    # Liczymy zmiany
    cpu_usage_diff = end_cpu_percent - start_cpu_percent
    memory_usage_diff = (end_memory_info - start_memory_info) / (1024 * 1024)  # MB

    if dpg.does_item_exist("BenchmarkWindow"):
        dpg.delete_item("BenchmarkWindow")

    # Tworzymy popup
    with dpg.window(label="Statystyki filtracji", tag='BenchmarkWindow'):
        apply_window_geometry("BenchmarkWindow", default_pos=(820, 10), default_size=(500, 250))

        dpg.add_text(f"Czas filtracji: {elapsed_time:.3f} sekund")
        dpg.add_text(f"Zmiana zużycia CPU: {cpu_usage_diff:.2f}%")
        dpg.add_text(f"Zmiana użycia RAM: {memory_usage_diff:+.2f} MB")

    # Resetuj zmienne
    filter_start_time = None
    start_cpu_percent = None
    start_memory_info = None

# THD mierzy zniekształcenia sygnału, które powstają jako harmoniczne — im niższy THD, tym mniej zniekształcony sygnał.
def calculate_thd(signal, fs, fundamental_freq=None):
    # FFT
    N = len(signal)
    yf = fft(signal)
    yf = np.abs(yf[:N // 2])  # tylko dodatnie częstotliwości
    freqs = fftfreq(N, 1 / fs)[:N // 2]

    # Znajdź składową podstawową
    if fundamental_freq is None:
        idx_f1 = np.argmax(yf)
    else:
        idx_f1 = np.argmin(np.abs(freqs - fundamental_freq))

    fundamental = yf[idx_f1]
    
    # Harmoniczne to wielokrotności składowej podstawowej
    harmonics = []
    for i in range(2, 6):  # 2. do 5. harmonicznej
        target_freq = i * freqs[idx_f1]
        idx = np.argmin(np.abs(freqs - target_freq))
        harmonics.append(yf[idx])

    harmonic_power = np.sum(np.square(harmonics))
    thd = np.sqrt(harmonic_power) / fundamental

    return thd * 100  # jako %


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

    # Dodajemy napis bezpośrednio do viewport
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
    size = dpg.get_item_width(tag), dpg.get_item_height(tag)
    window_config[tag] = {
        "x": pos[0], "y": pos[1], "width": size[0], "height": size[1]
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

# Wczytywanie pliku audio i pokazanie wykresów
def load_wav_file_callback(sender, app_data):
    global audio_file, audio_file_path, sampling_rate

    audio_file_path = app_data['file_path_name']
    sampling_rate, audio_data = read(audio_file_path)

    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]  # wybierz tylko lewy kanał (bez uśredniania)

    # Konwersja do floatów - potrzebune do później filtracji itd.
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0

    print("Loaded audio values:")
    print(audio_data)

    audio_file = audio_data
    print(f"Załadowano plik: {audio_file_path}, dane: {audio_file.shape}")

    call_create_audio_play_callback()
    show_plot_callback()
    show_spectrum_callback()

def load_original_wav_file_callback(sender, app_data):
    global audio_file_original

    audio_file_path = app_data['file_path_name']
    sampling_rate_original, data = read(audio_file_path)

    if len(data.shape) > 1:
        data = data[:, 0]  # wybierz tylko lewy kanał (bez uśredniania)

    # Konwersja do floatów - potrzebune do później filtracji itd.
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0

    print("Original loaded audio values:")
    print(data)
    print(sampling_rate_original)

    audio_file_original = data
    print(f"Załadowano plik: {audio_file_path}, dane: {audio_file_original.shape}")


###############################################################################################################################################################

#  __       __  __      __  __    __  _______   ________   ______   __      __         ______  
# /  |  _  /  |/  \    /  |/  |  /  |/       \ /        | /      \ /  \    /  |       /      \ 
# $$ | / \ $$ |$$  \  /$$/ $$ | /$$/ $$$$$$$  |$$$$$$$$/ /$$$$$$  |$$  \  /$$/       /$$$$$$  |
# $$ |/$  \$$ | $$  \/$$/  $$ |/$$/  $$ |__$$ |$$ |__    $$ \__$$/  $$  \/$$/        $$ |  $$ |
# $$ /$$$  $$ |  $$  $$/   $$  $$<   $$    $$< $$    |   $$      \   $$  $$/         $$ |  $$ |
# $$ $$/$$ $$ |   $$$$/    $$$$$  \  $$$$$$$  |$$$$$/     $$$$$$  |   $$$$/          $$ |  $$ |
# $$$$/  $$$$ |    $$ |    $$ |$$  \ $$ |  $$ |$$ |_____ /  \__$$ |    $$ |          $$ \__$$ |
# $$$/    $$$ |    $$ |    $$ | $$  |$$ |  $$ |$$       |$$    $$/     $$ |          $$    $$/ 
# $$/      $$/     $$/     $$/   $$/ $$/   $$/ $$$$$$$$/  $$$$$$/      $$/            $$$$$$/  
                                                                                             
###############################################################################################################################################################

AmplitudePlotHeight = -1
AmplitudePlotWidth = -1
SpectrumPlotHeight = -1
SpectrumPlotWidth = -1

def show_plot_callback():
    global audio_file
    if audio_file is None:
        print("Najpierw wczytaj plik!")
        return

    if not isinstance(audio_file, np.ndarray):
        print("Błąd: dane audio nie są numpy array!")
        return

    # Sprawdzenie stereo/mono
    if audio_file.ndim == 2:
        data_to_plot = audio_file[:, 0]
    else:
        data_to_plot = audio_file

    if data_to_plot.size == 0:
        print("Brak danych audio do wyświetlenia.")
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

        # Czasami X może być dłuższy niż Y przez zaokrąglenie — przycinamy
        min_len = min(len(downsampled_data), len(downsampled_x))
        downsampled_data = downsampled_data[:min_len]
        downsampled_x = downsampled_x[:min_len]

    # --- USUWANIE STAREGO OKNA ---
    if dpg.does_item_exist("AmplitudeWindow"):
        dpg.delete_item("AmplitudeWindow")

    # --- TWORZENIE NOWEGO OKNA Z WYKRESEM ---
    with dpg.window(label="Wykres Amplitudy", tag="AmplitudeWindow"):
        apply_window_geometry("AmplitudeWindow", default_pos=(820, 10), default_size=(500, 250))

        with dpg.plot(label="Amplituda w czasie", height=AmplitudePlotHeight, width=AmplitudePlotWidth):
            dpg.add_plot_axis(dpg.mvXAxis, label="Próbki")
            with dpg.plot_axis(dpg.mvYAxis, label="Amplituda") as y_axis:
                dpg.add_line_series(downsampled_x.tolist(), downsampled_data.tolist(), label="Amplituda", parent=y_axis)

def show_spectrum_callback():
    global audio_file, sampling_rate
    if audio_file is None:
        print("Najpierw wczytaj plik!")
        return

    # Jeśli dane są typu int16, przeskaluj do floatów -1..1 tylko na potrzeby wyświetlenia
    if audio_file.dtype == np.int16:
        data_to_plot = audio_file.astype(np.float32) / 32767.0
    else:
        data_to_plot = np.copy(audio_file)

    # Obsługa mono/stereo
    if data_to_plot.ndim == 2:
        data_to_plot = data_to_plot[:, 0]  # Wybieramy pierwszy kanał

    if data_to_plot.size == 0:
        print("Brak danych do wyświetlenia.")
        return

    # --- FFT ---
    spectrum = np.abs(np.fft.fft(data_to_plot))
    freqs = np.fft.fftfreq(len(data_to_plot), d=1.0 / sampling_rate)

    # Używamy tylko dodatnich częstotliwości
    half = len(freqs) // 2
    freqs = freqs[:half]
    spectrum = spectrum[:half]

    # --- DOWNSAMPLING --- (dla lepszej wydajności przy dużych plikach)
    max_points = 5000
    if len(freqs) > max_points:
        factor = len(freqs) // max_points
        spectrum = spectrum[::factor]
        freqs = freqs[::factor]

    # Upewniamy się, że X i Y mają tę samą długość
    min_len = min(len(freqs), len(spectrum))
    freqs = freqs[:min_len]
    spectrum = spectrum[:min_len]

    if dpg.does_item_exist("SpectrumWindow"):
        dpg.delete_item("SpectrumWindow")

    # --- TWORZENIE NOWEGO OKNA Z WYKRESEM ---
    with dpg.window(label="Widmo częstotliwości (FFT) przed filtracją", tag="SpectrumWindow"):
        apply_window_geometry("SpectrumWindow", default_pos=(820, 270), default_size=(500, 250))

        with dpg.plot(label="Widmo częstotliwości", height=SpectrumPlotHeight, width=SpectrumPlotWidth):
            dpg.add_plot_axis(dpg.mvXAxis, label="Częstotliwość [Hz]")
            with dpg.plot_axis(dpg.mvYAxis, label="Amplituda") as y_axis:
                dpg.add_line_series(freqs.tolist(), spectrum.tolist(), label="Widmo", parent=y_axis)


def show_original_plot_callback():
    global audio_file_original
    if audio_file_original is None:
        print("Najpierw wczytaj plik!")
        return

    if not isinstance(audio_file_original, np.ndarray):
        print("Błąd: dane audio nie są numpy array!")
        return

    # Sprawdzenie stereo/mono
    if audio_file_original.ndim == 2:
        data_to_plot = audio_file_original[:, 0]
    else:
        data_to_plot = audio_file_original

    if data_to_plot.size == 0:
        print("Brak danych audio do wyświetlenia.")
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

        # Czasami X może być dłuższy niż Y przez zaokrąglenie — przycinamy
        min_len = min(len(downsampled_data), len(downsampled_x))
        downsampled_data = downsampled_data[:min_len]
        downsampled_x = downsampled_x[:min_len]

    # --- USUWANIE STAREGO OKNA ---
    if dpg.does_item_exist("OriginalAmplitudeWindow"):
        dpg.delete_item("OriginalAmplitudeWindow")

    # --- TWORZENIE NOWEGO OKNA Z WYKRESEM ---
    with dpg.window(label="Wykres Amplitudy", tag="OriginalAmplitudeWindow"):
        apply_window_geometry("OriginalAmplitudeWindow", default_pos=(820, 10), default_size=(500, 250))
        with dpg.plot(label="Amplituda w czasie", height=AmplitudePlotHeight, width=AmplitudePlotWidth):
            dpg.add_plot_axis(dpg.mvXAxis, label="Próbki")
            with dpg.plot_axis(dpg.mvYAxis, label="Amplituda") as y_axis:
                line_tag = "orig_amp_line"
                dpg.add_line_series([], [], parent=y_axis)
                dpg.add_line_series([], [], parent=y_axis)
                dpg.add_line_series(downsampled_x.tolist(), downsampled_data.tolist(), label="Amplituda", parent=y_axis, tag=line_tag)


def apply_line_color(series_tag, color_rgba):
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvThemeCol_PlotLines, color_rgba, category=dpg.mvThemeCat_Plots)
    dpg.bind_item_theme(series_tag, theme)

def show_original_spectrum_callback():
    global audio_file_original, sampling_rate_original
    if audio_file_original is None:
        print("Najpierw wczytaj plik!")
        return

    # Jeśli dane są typu int16, przeskaluj do floatów -1..1 tylko na potrzeby wyświetlenia
    if audio_file_original.dtype == np.int16:
        data_to_plot = audio_file_original.astype(np.float32) / 32767.0
    else:
        data_to_plot = np.copy(audio_file_original)

    if sampling_rate_original is None:
        print("sampling_rate_original is None, setting 44100Hz")
        sampling_rate_original = 44100

    # Obsługa mono/stereo
    if data_to_plot.ndim == 2:
        data_to_plot = data_to_plot[:, 0]  # Wybieramy pierwszy kanał

    if data_to_plot.size == 0:
        print("Brak danych do wyświetlenia.")
        return

    # --- FFT ---
    spectrum = np.abs(np.fft.fft(data_to_plot))
    freqs = np.fft.fftfreq(len(data_to_plot), d=1.0 / sampling_rate_original)

    # Używamy tylko dodatnich częstotliwości
    half = len(freqs) // 2
    freqs = freqs[:half]
    spectrum = spectrum[:half]

    # --- DOWNSAMPLING --- (dla lepszej wydajności przy dużych plikach)
    max_points = 5000
    if len(freqs) > max_points:
        factor = len(freqs) // max_points
        spectrum = spectrum[::factor]
        freqs = freqs[::factor]

    # Upewniamy się, że X i Y mają tę samą długość
    min_len = min(len(freqs), len(spectrum))
    freqs = freqs[:min_len]
    spectrum = spectrum[:min_len]

    if dpg.does_item_exist("OriginalSpectrumWindow"):
        dpg.delete_item("OriginalSpectrumWindow")

    # --- TWORZENIE NOWEGO OKNA Z WYKRESEM ---
    with dpg.window(label="Widmo częstotliwości (FFT) przed filtracją", tag="OriginalSpectrumWindow"):
        apply_window_geometry("OriginalSpectrumWindow", default_pos=(820, 270), default_size=(500, 250))

        with dpg.plot(label="Widmo częstotliwości", height=SpectrumPlotHeight, width=SpectrumPlotWidth):
            dpg.add_plot_axis(dpg.mvXAxis, label="Częstotliwość [Hz]")
            with dpg.plot_axis(dpg.mvYAxis, label="Amplituda") as y_axis:
                line_tag = "orig_spec_line"
                dpg.add_line_series([], [], parent=y_axis)
                dpg.add_line_series([], [], parent=y_axis)
                dpg.add_line_series(freqs.tolist(), spectrum.tolist(), label="Widmo", parent=y_axis, tag=line_tag)


###############################################################################################################################################################

#  __       __  __      __  __    __  _______   ________   ______   __      __        ________ 
# /  |  _  /  |/  \    /  |/  |  /  |/       \ /        | /      \ /  \    /  |      /        |
# $$ | / \ $$ |$$  \  /$$/ $$ | /$$/ $$$$$$$  |$$$$$$$$/ /$$$$$$  |$$  \  /$$/       $$$$$$$$/ 
# $$ |/$  \$$ | $$  \/$$/  $$ |/$$/  $$ |__$$ |$$ |__    $$ \__$$/  $$  \/$$/        $$ |__    
# $$ /$$$  $$ |  $$  $$/   $$  $$<   $$    $$< $$    |   $$      \   $$  $$/         $$    |   
# $$ $$/$$ $$ |   $$$$/    $$$$$  \  $$$$$$$  |$$$$$/     $$$$$$  |   $$$$/          $$$$$/    
# $$$$/  $$$$ |    $$ |    $$ |$$  \ $$ |  $$ |$$ |_____ /  \__$$ |    $$ |          $$ |      
# $$$/    $$$ |    $$ |    $$ | $$  |$$ |  $$ |$$       |$$    $$/     $$ |          $$ |      
# $$/      $$/     $$/     $$/   $$/ $$/   $$/ $$$$$$$$/  $$$$$$/      $$/           $$/       
                                                                                             
###############################################################################################################################################################

AmplitudeFilteredPlotHeight = AmplitudePlotHeight
AmplitudeFilteredPlotWidth = AmplitudePlotWidth
SpectrumFilteredPlotHeight = SpectrumPlotHeight
SpectrumFilteredPlotWidth = SpectrumPlotWidth

def show_filtered_plot_callback():
    global audio_file_filtered
    if audio_file_filtered is None:
        print("Najpierw wczytaj plik!")
        return

    if not isinstance(audio_file_filtered, np.ndarray):
        print("Błąd: dane audio nie są numpy array!")
        return

    # Sprawdzenie stereo/mono
    if audio_file_filtered.ndim == 2:
        data_to_plot = audio_file_filtered[:, 0]
    else:
        data_to_plot = audio_file_filtered

    if data_to_plot.size == 0:
        print("Brak danych audio do wyświetlenia.")
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

        # Czasami X może być dłuższy niż Y przez zaokrąglenie — przycinamy
        min_len = min(len(downsampled_data), len(downsampled_x))
        downsampled_data = downsampled_data[:min_len]
        downsampled_x = downsampled_x[:min_len]

    # --- USUWANIE STAREGO OKNA ---
    if dpg.does_item_exist("AmplitudeWindowFiltered"):
        dpg.delete_item("AmplitudeWindowFiltered")

    # --- TWORZENIE NOWEGO OKNA Z WYKRESEM ---
    with dpg.window(label="Wykres Amplitudy", tag="AmplitudeWindowFiltered"):
        apply_window_geometry("AmplitudeWindowFiltered", default_pos=(820, 10), default_size=(500, 250))

        with dpg.plot(label="Amplituda w czasie", height=AmplitudePlotHeight, width=AmplitudePlotWidth):
            dpg.add_plot_axis(dpg.mvXAxis, label="Próbki")
            with dpg.plot_axis(dpg.mvYAxis, label="Amplituda") as y_axis:
                dpg.add_line_series(downsampled_x.tolist(), downsampled_data.tolist(), label="Amplituda", parent=y_axis)

def show_filtered_spectrum_callback():
    global audio_file_filtered, sampling_rate

    if audio_file_filtered is None:
        print("Najpierw wczytaj plik!")
        return

    # Jeśli dane są typu int16, przeskaluj do floatów -1..1 tylko na potrzeby wyświetlenia
    if audio_file_filtered.dtype == np.int16:
        data_to_plot = audio_file_filtered.astype(np.float32) / 32767.0
    else:
        data_to_plot = np.copy(audio_file_filtered)

    # Obsługa mono/stereo
    if data_to_plot.ndim == 2:
        data_to_plot = data_to_plot[:, 0]  # Wybieramy pierwszy kanał

    if data_to_plot.size == 0:
        print("Brak danych do wyświetlenia.")
        return

    # --- FFT ---
    spectrum = np.abs(np.fft.fft(data_to_plot))
    freqs = np.fft.fftfreq(len(data_to_plot), d=1.0 / sampling_rate)

    # Używamy tylko dodatnich częstotliwości
    half = len(freqs) // 2
    freqs = freqs[:half]
    spectrum = spectrum[:half]

    # --- DOWNSAMPLING --- (dla lepszej wydajności przy dużych plikach)
    max_points = 5000
    if len(freqs) > max_points:
        factor = len(freqs) // max_points
        spectrum = spectrum[::factor]
        freqs = freqs[::factor]

    # Upewniamy się, że X i Y mają tę samą długość
    min_len = min(len(freqs), len(spectrum))
    freqs = freqs[:min_len]
    spectrum = spectrum[:min_len]

    if dpg.does_item_exist("SpectrumWindowFiltered"):
        dpg.delete_item("SpectrumWindowFiltered")

    # --- TWORZENIE NOWEGO OKNA Z WYKRESEM ---
    with dpg.window(label="Widmo częstotliwości (FFT) po filtracji", tag="SpectrumWindowFiltered"):
        apply_window_geometry("SpectrumWindowFiltered", default_pos=(820, 270), default_size=(500, 250))

        with dpg.plot(label="Widmo częstotliwości", height=SpectrumFilteredPlotHeight, width=SpectrumFilteredPlotWidth):
            dpg.add_plot_axis(dpg.mvXAxis, label="Częstotliwość [Hz]")
            with dpg.plot_axis(dpg.mvYAxis, label="Amplituda") as y_axis:
                dpg.add_line_series(freqs.tolist(), spectrum.tolist(), label="Widmo", parent=y_axis)


###############################################################################################################################################################

#  ______  __    __  __    __  ________        __       __  __      __  __    __  _______   ________   ______   __      __ 
# /      |/  \  /  |/  \  /  |/        |      /  |  _  /  |/  \    /  |/  |  /  |/       \ /        | /      \ /  \    /  |
# $$$$$$/ $$  \ $$ |$$  \ $$ |$$$$$$$$/       $$ | / \ $$ |$$  \  /$$/ $$ | /$$/ $$$$$$$  |$$$$$$$$/ /$$$$$$  |$$  \  /$$/ 
#   $$ |  $$$  \$$ |$$$  \$$ |$$ |__          $$ |/$  \$$ | $$  \/$$/  $$ |/$$/  $$ |__$$ |$$ |__    $$ \__$$/  $$  \/$$/  
#   $$ |  $$$$  $$ |$$$$  $$ |$$    |         $$ /$$$  $$ |  $$  $$/   $$  $$<   $$    $$< $$    |   $$      \   $$  $$/   
#   $$ |  $$ $$ $$ |$$ $$ $$ |$$$$$/          $$ $$/$$ $$ |   $$$$/    $$$$$  \  $$$$$$$  |$$$$$/     $$$$$$  |   $$$$/    
#  _$$ |_ $$ |$$$$ |$$ |$$$$ |$$ |_____       $$$$/  $$$$ |    $$ |    $$ |$$  \ $$ |  $$ |$$ |_____ /  \__$$ |    $$ |    
# / $$   |$$ | $$$ |$$ | $$$ |$$       |      $$$/    $$$ |    $$ |    $$ | $$  |$$ |  $$ |$$       |$$    $$/     $$ |    
# $$$$$$/ $$/   $$/ $$/   $$/ $$$$$$$$/       $$/      $$/     $$/     $$/   $$/ $$/   $$/ $$$$$$$$/  $$$$$$/      $$/     
                                                                                                                         
###############################################################################################################################################################

BeforeAfterWindowAmplitudeWidth=-1
BeforeAfterWindowAmplitudeHeight=-1
BeforeAfterWindowSpectrumWidth=-1
BeforeAfterWindowSpectrumHeight=-1
DifferenceAmplitudeWindowWidth=-1
DifferenceAmplitudeWindowHeight=-1
DifferenceSpectrumWindowWidth=-1
DifferenceSpectrumWindowHeight=-1

def show_signal_difference():
    if audio_file is None or audio_file_filtered is None:
        print("Brak danych do porównania.")
        return

    if len(audio_file) != len(audio_file_filtered):
        print(f"Sygnały mają różne długości:" \
            f"audio_file: {len(audio_file)}," \
            f"audio_file_filtered: {len(audio_file_filtered)},")
        return

    # Downsampling dla płynności
    max_points = 10000
    step = max(1, len(audio_file) // max_points)
    x_data = np.arange(0, len(audio_file), step)
    y_orig = audio_file[::step]
    y_filt = audio_file_filtered[::step]

    # --- Różnica w dziedzinie czasu
    diff_signal = np.array(audio_file) - np.array(audio_file_filtered)
    max_points = 10000
    step = max(1, len(diff_signal) // max_points)
    x_data = np.arange(0, len(diff_signal), step)
    y_diff = diff_signal[::step]

    # --- Różnica w dziedzinie częstotliwości
    fft_orig = np.fft.fft(audio_file)
    fft_filt = np.fft.fft(audio_file_filtered)
    fft_diff = np.abs(fft_orig) - np.abs(fft_filt) 
    freqs = np.fft.fftfreq(len(audio_file), 1 / sampling_rate)
    
    half_n = len(freqs) // 2
    freqs = freqs[:half_n]
    fft_diff = fft_diff[:half_n]

    print(len(freqs))
    print(len(fft_orig))
    print(len(fft_filt))
    print(len(fft_diff))

    # half_n = min(len(freqs), len(fft_orig) // 2, len(fft_filt) // 2)
    # freqs_half = freqs[:half_n]
    fft_orig_half = np.abs(fft_orig[:half_n])
    fft_filt_half = np.abs(fft_filt[:half_n])

    window_tag = "BeforeAfterPlotWindow"
    if dpg.does_item_exist(window_tag):
        dpg.delete_item(window_tag)

    with dpg.window(label="Sygnał przed i po filtracji", tag=window_tag, width=BeforeAfterWindowAmplitudeWidth, height=BeforeAfterWindowAmplitudeHeight):
        apply_window_geometry(window_tag, default_pos=(820, 10), default_size=(600, 300))

        with dpg.plot(label="Amplituda w czasie", height=BeforeAfterWindowAmplitudeHeight, width=BeforeAfterWindowAmplitudeWidth):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="Próbki")
            with dpg.plot_axis(dpg.mvYAxis, label="Amplituda") as y_axis:
                dpg.add_line_series(x_data.tolist(), y_orig.tolist(), label="Oryginalny", parent=y_axis)
                dpg.add_line_series(x_data.tolist(), y_filt.tolist(), label="Po filtracji", parent=y_axis)

    window_tag = "DifferenceTimeDomainWindow"
    if dpg.does_item_exist(window_tag):
        dpg.delete_item(window_tag)

    with dpg.window(label="Różnica sygnałów w czasie", tag=window_tag, width=DifferenceAmplitudeWindowWidth, height=DifferenceAmplitudeWindowHeight):
        apply_window_geometry(window_tag, default_pos=(820, 330), default_size=(600, 300))

        with dpg.plot(label="Różnica amplitudy", height=DifferenceAmplitudeWindowHeight, width=DifferenceAmplitudeWindowWidth):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="Próbki")
            with dpg.plot_axis(dpg.mvYAxis, label="Różnica amplitud (oryginal - filtered)") as y_axis:
                dpg.add_line_series(x_data.tolist(), y_diff.tolist(), label="Różnica", parent=y_axis)

    window_tag = "BeforeAfterFrequencyDomainWindow"
    if dpg.does_item_exist(window_tag):
        dpg.delete_item(window_tag)

    with dpg.window(label="Widmo przed i po filtracji", tag=window_tag, width=BeforeAfterWindowAmplitudeWidth, height=BeforeAfterWindowAmplitudeHeight):
        apply_window_geometry(window_tag, default_pos=(820, 10), default_size=(600, 300))

        with dpg.plot(label="Widmo przed i po filtracji", height=BeforeAfterWindowAmplitudeHeight, width=BeforeAfterWindowAmplitudeWidth):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="Częstotliwość [Hz]")
            with dpg.plot_axis(dpg.mvYAxis, label="Amplituda") as y_axis:
                dpg.add_line_series(freqs.tolist(), fft_orig_half.tolist(), label="Oryginalne widmo", parent=y_axis)
                dpg.add_line_series(freqs.tolist(), fft_filt_half.tolist(), label="Widmo po filtracji", parent=y_axis)

    spec_window_tag = "DifferenceFrequencyDomainWindow"
    if dpg.does_item_exist(spec_window_tag):
        dpg.delete_item(spec_window_tag)

    with dpg.window(label="Różnica widmowa", tag=spec_window_tag, width=DifferenceSpectrumWindowWidth, height=DifferenceSpectrumWindowHeight):
        apply_window_geometry(spec_window_tag, default_pos=(820, 660), default_size=(600, 300))

        with dpg.plot(label="Różnica widm", height=DifferenceSpectrumWindowHeight, width=DifferenceSpectrumWindowWidth):
            dpg.add_plot_axis(dpg.mvXAxis, label="Częstotliwość [Hz]")
            with dpg.plot_axis(dpg.mvYAxis, label="Różnica widm (oryginal - filtered)") as y_axis:
                dpg.add_line_series(freqs.tolist(), fft_diff.tolist(), label="Δ widmo", parent=y_axis)



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
    global audio_file, audio_file_filtered, sampling_rate

    if audio_file is None:
        print("Najpierw wczytaj plik!")
        return

    filter_type = dpg.get_value("low_filter_type")
    filter_order = int(dpg.get_value("low_filter_order"))
    cutoff_freq = int(dpg.get_value("low_cutoff_freq"))
    new_sampling_rate = int(dpg.get_value("low_sampling_rate"))

    if cutoff_freq <= 0 or cutoff_freq >= 0.5 * new_sampling_rate:
        print("Niepoprawna częstotliwość odcięcia!")
        return

    nyquist = 0.5 * new_sampling_rate
    normalized_cutoff = cutoff_freq / nyquist

    start_benchmark()
    start_processing_indicator()

    # Wybór filtra
    if filter_type == "Butterworth":
        b, a = butter(filter_order, normalized_cutoff, btype='low', analog=False)
    elif filter_type == "Chebyshev":
        b, a = cheby1(filter_order, 1, normalized_cutoff, btype='low', analog=False)
    elif filter_type == "Bessel":
        b, a = bessel(filter_order, normalized_cutoff, btype='low', analog=False, norm='phase')
    else:
        print("Nieznany typ filtra!")
        return

    # # Filtrujemy każdy kanał osobno
    if audio_file.ndim == 2:
        for channel in range(audio_file.shape[1]):
            audio_file_filtered[:, channel] = filtfilt(b, a, audio_file[:, channel])
    else:
        audio_file_filtered = filtfilt(b, a, audio_file)

    stop_benchmark_and_show_results()
    stop_processing_indicator()

    thd_before = calculate_thd(audio_file, sampling_rate)
    thd_low = calculate_thd(audio_file_filtered, new_sampling_rate)

    file_original_path = Path(audio_file_path)
    file_new_path = file_original_path.with_stem(f"{Path(audio_file_path).stem}_LOW_{filter_type}_{filter_order}_{cutoff_freq}")
    
    print(f"Zastosowano filtr: {filter_type}, rząd: {filter_order}, odcięcie: {cutoff_freq} Hz, THD before: {thd_before} THD after: {thd_low}")

    show_filtered_plot_callback()
    show_filtered_spectrum_callback()
    call_create_audio_filtered_play_callback()

    print(f"Zapisywanie do pliku: {file_new_path}")
    save_audio_with_convert(file_new_path, new_sampling_rate, audio_file_filtered)
    print(f"Zapisano do pliku: {file_new_path}")
    dpg.hide_item("lowpass_filter_popup")

# Okno Popup Filtracji
with dpg.window(label="Filtracja dolnoprzepustowa", tag="lowpass_filter_popup", modal=True, show=False, width=800, height=200):
    dpg.add_combo(("Butterworth", "Chebyshev", "Bessel"), label="Typ filtra", tag="low_filter_type", default_value="Butterworth")
    dpg.add_input_int(label="Rząd filtra", tag="low_filter_order", default_value=4, min_value=1, max_value=10)
    dpg.add_input_int(label="Częstotliwość odcięcia (Hz)", tag="low_cutoff_freq", default_value=5000, min_value=1, max_value=22000)
    dpg.add_input_int(label="Częstotliwość próbkowania (Hz)", tag="low_sampling_rate", default_value=44100, min_value=1000, max_value=96000)
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
    global audio_file, audio_file_filtered, sampling_rate

    if audio_file is None:
        print("Najpierw wczytaj plik!")
        return

    filter_type = dpg.get_value("hi_filter_type")
    filter_order = int(dpg.get_value("hi_filter_order"))
    cutoff_freq = int(dpg.get_value("hi_cutoff_freq"))
    new_sampling_rate = int(dpg.get_value("hi_sampling_rate"))

    if cutoff_freq <= 0 or cutoff_freq >= 0.5 * new_sampling_rate:
        print("Niepoprawna częstotliwość odcięcia!")
        return

    nyquist = 0.5 * new_sampling_rate
    normalized_cutoff = cutoff_freq / nyquist

    start_benchmark()
    start_processing_indicator()

    # Wybór filtra
    if filter_type == "Butterworth":
        b, a = butter(filter_order, normalized_cutoff, btype='high', analog=False)
    elif filter_type == "Chebyshev":
        b, a = cheby1(filter_order, 1, normalized_cutoff, btype='high', analog=False)
    elif filter_type == "Bessel":
        b, a = bessel(filter_order, normalized_cutoff, btype='high', analog=False, norm='phase')
    else:
        print("Nieznany typ filtra!")
        return

    # # Filtrujemy każdy kanał osobno
    if audio_file.ndim == 2:
        for channel in range(audio_file.shape[1]):
            audio_file_filtered[:, channel] = filtfilt(b, a, audio_file[:, channel])
    else:
        audio_file_filtered = filtfilt(b, a, audio_file)

    stop_benchmark_and_show_results()
    stop_processing_indicator()

    thd_before = calculate_thd(audio_file, sampling_rate)
    thd_high = calculate_thd(audio_file_filtered, new_sampling_rate)

    print(f"Zastosowano filtr: {filter_type}, rząd: {filter_order}, odcięcie: {cutoff_freq} Hz, THD before: {thd_before}, THD: {thd_high}")

    file_original_path = Path(audio_file_path)
    file_new_path = file_original_path.with_stem(f"{Path(audio_file_path).stem}_HIGH_{filter_type}_{filter_order}_{cutoff_freq}")
    
    show_filtered_plot_callback()
    show_filtered_spectrum_callback()
    call_create_audio_filtered_play_callback()

    print(f"Zapisywanie do pliku: {file_new_path}")
    save_audio_with_convert(file_new_path, new_sampling_rate, audio_file_filtered)
    print(f"Zapisano do pliku: {file_new_path}")
    dpg.hide_item("hipass_filter_popup")

# Okno Popup Filtracji
with dpg.window(label="Filtracja górnoprzepustowa", tag="hipass_filter_popup", modal=True, show=False, width=800, height=200):
    dpg.add_combo(("Butterworth", "Chebyshev", "Bessel"), label="Typ filtra", tag="hi_filter_type", default_value="Butterworth")
    dpg.add_input_int(label="Rząd filtra", tag="hi_filter_order", default_value=4, min_value=1, max_value=10)
    dpg.add_input_int(label="Częstotliwość odcięcia (Hz)", tag="hi_cutoff_freq", default_value=5000, min_value=1, max_value=22000)
    dpg.add_input_int(label="Częstotliwość próbkowania (Hz)", tag="hi_sampling_rate", default_value=44100, min_value=1000, max_value=96000)
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
    global audio_file, audio_file_filtered, sampling_rate

    if audio_file is None:
        print("Najpierw wczytaj plik!")
        return

    filter_type = dpg.get_value("band_filter_type")
    filter_order = int(dpg.get_value("band_filter_order"))
    cutoff_low = int(dpg.get_value("band_cutoff_freq_low"))
    cutoff_high = int(dpg.get_value("band_cutoff_freq_high"))
    new_sampling_rate = int(dpg.get_value("band_sampling_rate"))

    if cutoff_low <= 0 or cutoff_high <= cutoff_low or cutoff_high >= new_sampling_rate / 2:
        print("Niepoprawne częstotliwości odcięcia!")
        return

    nyquist = new_sampling_rate / 2
    normalized_cutoff = [cutoff_low / nyquist, cutoff_high / nyquist]

    start_benchmark()
    start_processing_indicator()

    # Wybór filtra
    if filter_type == "Butterworth":
        b, a = butter(filter_order, normalized_cutoff, btype='band', analog=False)
    elif filter_type == "Chebyshev":
        b, a = cheby1(filter_order, 1, normalized_cutoff, btype='band', analog=False)
    elif filter_type == "Bessel":
        b, a = bessel(filter_order, normalized_cutoff, btype='band', analog=False, norm='phase')
    else:
        print("Nieznany typ filtra!")
        return

    # Filtrujemy każdy kanał osobno
    if audio_file.ndim == 2:
        for channel in range(audio_file.shape[1]):
            audio_file_filtered[:, channel] = filtfilt(b, a, audio_file[:, channel])
    else:
        audio_file_filtered = filtfilt(b, a, audio_file)

    stop_benchmark_and_show_results()
    stop_processing_indicator()

    thd_before = calculate_thd(audio_file, sampling_rate) 
    thd_band = calculate_thd(audio_file_filtered, new_sampling_rate)

    print(f"Zastosowano filtr: {filter_type}, rząd: {filter_order}, odcięcie dolne: {cutoff_low}, odcięcie górne: {cutoff_high} Hz, THD before: {thd_before}, THD: {thd_band}")

    file_original_path = Path(audio_file_path)
    file_new_path = file_original_path.with_stem(f"{Path(audio_file_path).stem}_BAND_{filter_type}_{filter_order}_{cutoff_low}-{cutoff_high}")
    
    show_filtered_plot_callback()
    show_filtered_spectrum_callback()
    call_create_audio_filtered_play_callback()

    print(f"Zapisywanie do pliku: {file_new_path}")
    save_audio_with_convert(file_new_path, new_sampling_rate, audio_file_filtered)
    print(f"Zapisano do pliku: {file_new_path}")
    dpg.hide_item("bandpass_filter_popup")

# Okno Popup Filtracji
with dpg.window(label="Filtracja pasmowoprzepustowa", tag="bandpass_filter_popup", modal=True, show=False, width=800, height=250):
    dpg.add_combo(("Butterworth", "Chebyshev", "Bessel"), label="Typ filtra", tag="band_filter_type", default_value="Butterworth")
    dpg.add_input_int(label="Rząd filtra", tag="band_filter_order", default_value=4, min_value=1, max_value=10)
    dpg.add_input_int(label="Dolna częstotliwość odcięcia (Hz)", tag="band_cutoff_freq_low", default_value=300, min_value=1, max_value=22000)
    dpg.add_input_int(label="Górna częstotliwość odcięcia (Hz)", tag="band_cutoff_freq_high", default_value=5000, min_value=1, max_value=22000)
    dpg.add_input_int(label="Częstotliwość próbkowania (Hz)", tag="band_sampling_rate", default_value=44100, min_value=1000, max_value=96000)
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

def lms_filter(x, d, filter_order=32, mu=0.001):
    """
    x - sygnał wejściowy
    d - sygnał odniesienia (może być x lub inny)
    filter_order - liczba współczynników filtra
    mu - współczynnik uczenia
    """
    # min_length = min(len(x), len(d))
    # x = np.array(x[:min_length])
    # d = np.array(d[:min_length])

    # Konwersja danych do formatu wymaganego przez padasip
    x_matrix = pa.input_from_history(x, filter_order)  # Kształt (N, filter_order)
    d_vector = np.array(d[filter_order-1:])  # Obcięcie do zgodnych wymiarów

    # Inicjalizacja i uruchomienie filtra
    f = pa.filters.FilterLMS(n=filter_order, mu=mu, w="zeros")
    y, e, w = f.run(d_vector, x_matrix)  # Uwaga: kolejność (d, x)!
    
    return y, e, w

def measure_lms_convergence(e, stable_duration=10, threshold_ratio=1.05):
    """
    e - wektor błędu z LMS
    stable_duration - liczba kolejnych próbek, przez które błąd musi być stabilny
    threshold_ratio - jak blisko musi być do minimalnego błędu
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
    global audio_file, audio_file_filtered 

    if audio_file is None:
        print("Najpierw wczytaj plik!")
        return

    filter_length = int(dpg.get_value("lms_filter_length"))
    mu = float(dpg.get_value("lms_learning_rate"))

    start_benchmark()
    start_processing_indicator()

    if audio_file.ndim == 2:
        for channel in range(audio_file.shape[1]):
            audio_file_filtered[:, channel], error, weights = lms_filter(audio_file[:, channel], audio_file[:, channel], filter_length, mu)
    else:
        audio_file_filtered, error, weights  = lms_filter(audio_file, audio_file, filter_length, mu)

    convergence_index = measure_lms_convergence(error)
    print(f"Szybkość konwergencji: {convergence_index} próbek")

    max_len = max(len(audio_file), len(audio_file_filtered))
    audio_file_filtered = np.pad(audio_file_filtered, (max_len - len(audio_file_filtered), 0), mode='constant')

    stop_benchmark_and_show_results()
    stop_processing_indicator()

    file_original_path = Path(audio_file_path)
    file_new_path = file_original_path.with_stem(f"{Path(audio_file_path).stem}_LMS_{filter_length}_{mu}")
    
    show_filtered_plot_callback()
    show_filtered_spectrum_callback()
    call_create_audio_filtered_play_callback()

    print(f"Zastosowano filtr LMS: długość = {filter_length}, współczynnik uczenia = {mu}")
    print(f"Zapisywanie do pliku: {file_new_path}")
    save_audio_with_convert(file_new_path, sampling_rate, audio_file_filtered)
    print(f"Zapisano do pliku: {file_new_path}")
    dpg.hide_item("lms_filter_popup")

# Okno Popup Filtracji
with dpg.window(label="Filtracja LMS", tag="lms_filter_popup", modal=True, show=False, width=800, height=300):

    dpg.add_input_int(label="Długość filtra", tag="lms_filter_length", default_value=32, min_value=1, step=1, max_value=1024)
    dpg.add_input_float(label="Współczynnik uczenia (μ)", tag="lms_learning_rate", default_value=0.001, step=0.001, min_value=0.00001, max_value=1.0)
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
    
    # Thresholding: zerowanie małych współczynników
    new_coeffs = []
    for c in coeffs:
        c = pywt.threshold(c, threshold * np.max(c), mode='soft')
        new_coeffs.append(c)
    
    reconstructed_signal = pywt.waverec(new_coeffs, wavelet_name)

    # Dopasuj długość sygnału
    return reconstructed_signal[:len(signal)]

def open_WAVELET_filter_popup_callback():
    dpg.show_item("wavelet_filter_popup")

def apply_WAVELET_filter_callback():
    global audio_file, audio_file_filtered

    if audio_file is None:
        print("Najpierw wczytaj plik!")
        return

    wavelet_name = dpg.get_value("wavelet_name")
    decomposition_level = int(dpg.get_value("wavelet_level"))
    threshold = float(dpg.get_value("wavelet_threshold"))

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
    file_new_path = file_original_path.with_stem(f"{Path(audio_file_path).stem}_WAVELET_{wavelet_name}_{decomposition_level}_{threshold}")
        
    show_filtered_plot_callback()
    show_filtered_spectrum_callback()
    call_create_audio_filtered_play_callback()

    print(f"Zastosowano filtrację wavelet: {wavelet_name}, poziom: {decomposition_level}, próg: {threshold}")
    print(f"Zapisywanie do pliku: {file_new_path}")
    save_audio_with_convert(file_new_path, sampling_rate, audio_file_filtered)
    print(f"Zapisano do pliku: {file_new_path}")
    dpg.hide_item("wavelet_filter_popup")

# Okno Popup Filtracji
with dpg.window(label="Filtracja Wavelet", tag="wavelet_filter_popup", modal=True, show=False, width=800, height=400):
    dpg.add_combo(("db4", "haar", "sym5", "coif1"), label="Typ falki", tag="wavelet_name", default_value="db4")
    dpg.add_input_int(label="Poziom dekompozycji", tag="wavelet_level", default_value=4, step=1, min_value=1, max_value=10)
    dpg.add_input_float(label="Próg (threshold)", tag="wavelet_threshold", default_value=0.04, step=0.01, min_value=0.001, max_value=1.0)
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
    z - wejściowy sygnał (obserwacje)
    Q - szum procesu
    R - szum pomiaru
    """
    n_iter = len(z)
    sz = (n_iter,)

    # Inicjalizacja zmiennych
    xhat = np.zeros(sz)      # Estymata a posteriori
    P = np.zeros(sz)         # Błąd estymacji a posteriori
    xhatminus = np.zeros(sz) # Estymata a priori
    Pminus = np.zeros(sz)    # Błąd estymacji a priori
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
    global audio_file, audio_file_filtered

    if audio_file is None:
        print("Najpierw wczytaj plik!")
        return

    Q = float(dpg.get_value("kalman_Q"))
    R = float(dpg.get_value("kalman_R"))

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
    file_new_path = file_original_path.with_stem(f"{Path(audio_file_path).stem}_KALMAN_{Q}_{R}")
    
    show_filtered_plot_callback()
    show_filtered_spectrum_callback()
    call_create_audio_filtered_play_callback()

    print(f"Zastosowano filtrację Kalmana: Q={Q}, R={R}")
    print(f"Zapisywanie do pliku: {file_new_path}")
    save_audio_with_convert(file_new_path, sampling_rate, audio_file_filtered)
    print(f"Zapisano do pliku: {file_new_path}")
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
    Wczytuje plik WAV lub zwraca podaną tablicę.
    Jeśli podano ścieżkę -> zwraca (y, sr), gdzie y.shape = (channels, samples).
    Jeśli podano ndarray:
      -      jeśli kształt (n,) -> mono
      -      jeśli kształt (n,2) -> stereo
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
            # zakładamy (channels, samples)
            return arr, sr or 0
        else:
            raise ValueError("Tablica musi mieć wymiar 1D (mono) lub 2D (stereo).")
    else:
        raise TypeError("path_or_array musi być ścieżką (str) lub ndarray.")

def calculate_snr2(
    original: Union[str, np.ndarray],
    noisy: Union[str, np.ndarray],
    sr: int = None,
    mode: str = 'mean'
) -> float:
    """
    Oblicza SNR (dB) między sygnałem oryginalnym a zaszumionym/po filtracji.

    Parametry:
    - original: ścieżka do WAV lub ndarray (mono lub stereo)
    - noisy:    ścieżka do WAV lub ndarray (mono lub stereo)
    - sr:       żądana częstotliwość próbkowania (tylko przy ndarray=None)
    - mode:     jak zredukować stereo do jednej wartości:
                'mean'  - średnia SNR z kanałów
                'mono'  - najpierw miksuje do mono (średnia kanałów), potem liczy
                'perch' - zwraca listę wartości [snr_ch0, snr_ch1, ...]

    Zwraca:
    - float (SNR w dB) albo listę floatów jeśli mode='perch'
    """
    # Wczytanie
    sig_orig, sr1 = load_audio(original, sr)
    sig_noisy, sr2 = load_audio(noisy,    sr)
    if sr1 and sr2 and sr1 != sr2:
        raise ValueError(f"Różne sr: {sr1} vs {sr2}")
    
    # Przytnij do tej samej długości
    n = min(sig_orig.shape[1], sig_noisy.shape[1])
    sig_orig = sig_orig[:, :n]
    sig_noisy = sig_noisy[:, :n]

    # Oblicz szum
    noise = sig_noisy - sig_orig

    # Moc sygnału i mocy szumu per kanał
    p_signal = np.mean(sig_orig**2, axis=1)
    p_noise  = np.mean(noise**2,    axis=1)

    # Unikamy dzielenia przez zero
    p_noise = np.where(p_noise == 0, np.finfo(float).eps, p_noise)

    # SNR per kanał (linia)
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

# — przykład użycia —

# # 1) Z plików WAV:
# snr_db = calculate_snr("original.wav", "noisy.wav", mode='mean')
# print(f"SNR (średnie): {snr_db:.2f} dB")

# 2) Z tablic NumPy:
#    audio_orig = np.array([...])
#    audio_noisy = np.array([...])
#    snr_db = calculate_snr(audio_orig, audio_noisy, sr=44100, mode='perch')
#    print("SNR per channel:", snr_db)


# Funkcja obliczająca kalsyczny zwykły snr
def calculate_snr(original, noisy_or_filtered):
    noise = noisy_or_filtered - original

    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float('inf')  # Idealny przypadek
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# Funkcja obliczająca segmentowy snr
def compute_segmental_snr(original_signal, noisy_or_filtered, sampling_rate, frame_duration_ms=20):
    original_signal = original_signal.astype(np.float32)
    noisy_or_filtered = noisy_or_filtered.astype(np.float32)

    # Jeśli dane są w zakresie int16, znormalizuj:
    if np.max(np.abs(original_signal)) > 1.0:
        original_signal /= 32768.0
    if np.max(np.abs(noisy_or_filtered)) > 1.0:
        noisy_or_filtered /= 32768.0

    if original_signal.shape != noisy_or_filtered.shape:
        raise ValueError("Sygnały muszą mieć tą samą długość")

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

# Funkcja obliczająca SNR ważony krytycznie (Critical-band SNR) – uwzględnia, w jakim paśmie słuch jest wrażliwszy.
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
        print(f"Błąd podczas oblicznia compute_critical_band_snr: {e}")
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

# Funkcja wywołujaca obliczanie SNR i wyswetlajaca wynik
def show_snr_analysis():
    global audio_file_original, audio_file, sampling_rate

    if audio_file_original is None:
        print("Nie wczytano oryginalnego (niezaszumionego) sygnału!")
        return

    if audio_file is None or audio_file_filtered is None:
        print("Brak danych: załaduj plik wejściowy i przefiltruj go.")
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

    print(f"SNR przed filtracją: {snr_before:.2f} dB")
    print(f"SNR po filtracji:   {snr_after:.2f} dB")
    
    print(f"SNR2 przed filtracją: {snr_bf:.2f} dB")
    print(f"SNR2 po filtracji:   {snr_af:.2f} dB")

    print(f"SNR segmental przed filtracją: {snr_segmental_before:.2f} dB")
    print(f"SNR segmental po filtracji:   {snr_segmental_after:.2f} dB")

    print(f"SNR CB przed filtracją: {snr_cb_before:.2f} dB")
    print(f"SNR CB po filtracji:   {snr_cb_after:.2f} dB")

    dpg.show_item("SNRResultWindow")
    dpg.set_value("SNRText", 
                  f"SNR przed filtracją: {snr_before:.2f} dB\nSNR po filtracji: {snr_after:.2f} dB\n\n" +
                  f"SNR2 przed filtracją: {snr_bf:.2f} dB\nSNR2 po filtracji: {snr_af:.2f} dB\n\n" +
                  f"SNR segmental przed filtracją: {snr_segmental_before:.2f} dB\nSNR segmental po filtracji: {snr_segmental_after:.2f} dB\n\n" +
                  f"SNR CB przed filtracją: {snr_cb_before:.2f} dB\nSNR CB po filtracji: {snr_cb_after:.2f} dB")


with dpg.window(label="SNR Analiza", tag="SNRResultWindow", show=False, width=400, height=150):
    apply_window_geometry("SNRResultWindow", default_pos=(0, 672), default_size=(400, 150))
    dpg.add_text("", tag="SNRText")

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

# Zapis przefiltrowanego sygnału do pliku
def save_audio(file_path, sampling_freq, data):
    write(file_path, sampling_freq, data.astype(np.int16))

def save_audio_with_convert(file_path, sampling_freq, data):

    if isinstance(sampling_freq, float):
        sampling_freq = int(sampling_freq)  # Konwersja na int

    if data.dtype != np.int16:
        data = np.clip(data, -1.0, 1.0)  # Upewniamy się że wartości są w zakresie
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
            print(f"Błąd w monitor_progress: {e}")
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
        print("Błąd: Brak danych audio!")
        return

    if dpg.does_item_exist(windowTag):
        dpg.delete_item(windowTag)

    with dpg.window(tag=windowTag, label=label):
        apply_window_geometry(windowTag, default_pos=(820, 270), default_size=(500, 250))

        with dpg.group():
            with dpg.group(horizontal=True):
                dpg.add_button(label="Start", callback=lambda: start_playback(audio_data, sampling_rate, label, windowTag))
                dpg.add_button(label="Stop", width=120, callback=lambda: stop_playback(audio_data))

            dpg.add_progress_bar(label="Postęp", tag=f"{windowTag}_progress_bar", width=480)
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
    frequencies: float lub lista floatów [Hz]
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
        print("Błąd: podano nieprawidłowe częstotliwości")
        return

    signal = generate_sine_wave(frequencies, duration, amplitude, sr)
    save_generated_signal(signal, sr, filename)

def open_gen_signal_window():
    if dpg.does_item_exist("SinWaveGeneratorWindow"):
        dpg.delete_item("SinWaveGeneratorWindow")

    with dpg.window(label="Generator sygnału", tag="SinWaveGeneratorWindow"):
        apply_window_geometry("SinWaveGeneratorWindow", default_pos=(820, 270), default_size=(500, 250))
        dpg.add_input_text(label="Częstotliwości [Hz] (np. 440 lub 440,880)", tag="gen_freq")
        dpg.add_input_float(label="Czas trwania [s]", default_value=2.0, tag="gen_duration")
        dpg.add_input_float(label="Amplituda", default_value=0.5, min_value=0.0, max_value=1.0, tag="gen_amp")
        dpg.add_input_int(label="Częstotliwość próbkowania", default_value=44100, tag="gen_sr")
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
        raise ValueError(f"Nieobsługiwany typ szumu: {noise_type}")

    # Skalowanie do żądanego SNR
    power_noise = np.mean(noise**2)
    target_noise_power = power_signal / (10 ** (snr_db / 10))
    scaling_factor = np.sqrt(target_noise_power / (power_noise + 1e-8))
    noise_scaled = noise * scaling_factor

    return signal + noise_scaled

def add_noise_callback():
    global audio_file, audio_file_noisy

    if audio_file is None:
        print("Najpierw wczytaj sygnał!")
        return

    sampling_ratee = 44100
    noise_type = dpg.get_value("noise_type")
    snr_value = float(dpg.get_value("snr_value"))
    noised_filename = dpg.get_value("noised_filename")

    audio_file_noisy = add_noise(audio_file, noise_type=noise_type, snr_db=snr_value, sampling_rate=sampling_ratee)

    print(f"Szum typu '{noise_type}' dodany z SNR = {snr_value} dB.")
    save_audio_with_convert(noised_filename, sampling_ratee, audio_file_noisy )

def open_add_noise_window_callback():
    if dpg.does_item_exist("AddNoiseWindow"):
        dpg.delete_item("AddNoiseWindow")

    with dpg.window(label="Dodaj szum", tag="AddNoiseWindow"):
        apply_window_geometry("AddNoiseWindow", default_pos=(0, 0), default_size=(500, 250))
        dpg.add_combo(label="Rodzaj szumu", items=["white", "pink", "urban", "industrial", "impulse"], default_value="white", tag="noise_type")
        dpg.add_slider_float(label="SNR [dB]", default_value=10.0, min_value=-10.0, max_value=80.0, tag="snr_value")
        dpg.add_input_text(label="Nazwa pliku", default_value="noised.wav", tag="noised_filename")
        dpg.add_button(label="Zaszum sygnał", callback=add_noise_callback)


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

# # Wczytaj plik z sygnałem i plik z szumem
# signal_waveform, sr = torchaudio.load(os.path.join(CURRENT_DIR, 'atlas_ATC_around_the_world.wav'))
# noise_waveform, sr_noise = torchaudio.load(os.path.join(CURRENT_DIR, 'atlas_ATC_around_the_world_noised_Pink.wav'))
# filtered_waveform, sr_filtered_noise = torchaudio.load(os.path.join(CURRENT_DIR, 'atlas_ATC_around_the_world_noised_Pink_BAND_Butterworth_4_300-5000.wav'))

# # Upewnij się, że oba sygnały mają tę samą długość
# min_length = min(signal_waveform.shape[1], noise_waveform.shape[1],  filtered_waveform.shape[1])
# signal_waveform = signal_waveform[:, :min_length]
# noise_waveform = noise_waveform[:, :min_length]
# filtered_waveform = filtered_waveform[:, :min_length]

# filtered_waveform = filtered_waveform.repeat(2, 1)

# # Oblicz SNR
# snr_value = signal_noise_ratio(signal_waveform, noise_waveform)
# snr_filtered_value = signal_noise_ratio(signal_waveform, filtered_waveform)

# # Średnia po kanałach
# snr_mean = snr_value.mean().item()
# print(f"Średnie snr_value SNR (stereo): {snr_mean:.2f} dB")

# snr_mean_filtered = snr_filtered_value.mean().item()
# print(f"Średnie snr_filtered_value SNR (stereo): {snr_mean_filtered:.2f} dB")


tableHeight = 760
tableWidth = 760

tracked_tags = [
    "MainWindow",
    "AmplitudeWindow",
    "AmplitudeWindowFiltered",
    "SpectrumWindow",
    "SpectrumWindowFiltered",
    "InputSignalPlaybackWindow",
    "FilteredSignalPlaybackWindow",
    "BeforeAfterPlotWindow",
    "DifferenceTimeDomainWindow",
    "DifferenceFrequencyDomainWindow",
    "BeforeAfterFrequencyDomainWindow",
    "SNRResultWindow",
    "SinWaveGeneratorWindow",
    "AddNoiseWindow",
    "OriginalAmplitudeWindow",
    "OriginalSpectrumWindow",
]

load_window_config()

def save_window_config_callback():
    for tag in tracked_tags: 
        if dpg.does_item_exist(tag):
            store_geometry_on_close("save_layout", None, tag) 

    save_window_config()


def call_create_audio_play_callback():
    create_audio_controls_window("InputSignalPlaybackWindow", "Odtwarzanie wczytanego pliku", audio_file, sampling_rate, "input")

def call_create_audio_filtered_play_callback():
    create_audio_controls_window("FilteredSignalPlaybackWindow", "Odtwarzanie przefiltrowanego pliku", audio_file_filtered, sampling_rate_filtered if sampling_rate_filtered is not None else sampling_rate, "filtered")

def open_load_dialog():
    dpg.show_item("load_wav_file_dialog")

def open_load_original_dialog():
    dpg.show_item("load_original_wav_file_dialog")

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

with dpg.window(label="Main Window", tag="MainWindow", no_resize=False):    
    apply_window_geometry("MainWindow", default_pos=(0, 0), default_size=(tableWidth, tableHeight))
    dpg.maximize_viewport()
    with dpg.group(horizontal=True):
        with dpg.child_window(autosize_y=True):
            dpg.add_button(label="Wczytaj plik oryginal (.wav)", callback=open_load_original_dialog)
            dpg.add_button(label="Wczytaj plik do filtracji (.wav)", callback=open_load_dialog)

            with dpg.collapsing_header(label="Filtry", default_open=True):
                dpg.add_button(label="Filtracja dolnoprzepustowa", callback=open_LOWPASS_filter_popup_callback)
                dpg.add_button(label="Filtracja górnoprzepustowa", callback=open_HIPASS_filter_popup_callback)
                dpg.add_button(label="Filtracja pasmoprzepustowa", callback=open_BANDPASS_filter_popup_callback)
                dpg.add_button(label="Filtracja LMS", callback=open_LMS_filter_popup_callback)
                dpg.add_button(label="Filtracja falkowa", callback=open_WAVELET_filter_popup_callback)
                dpg.add_button(label="Filtracja kalmana", callback=open_KALMAN_filter_popup_callback)
                # dpg.add_button(label="Filtracja AI", callback=open_AI_filter_popup_callback)

            with dpg.collapsing_header(label="Wykresy", default_open=True):
                dpg.add_button(label="Pokaż różnicę sygnałów", callback=show_signal_difference)

                dpg.add_button(label="Pokaz wykres amplitudy", callback=show_plot_callback)
                dpg.add_button(label="Pokaz wykres częstotliwości", callback=show_spectrum_callback)

                dpg.add_button(label="Pokaz wykres amplitudy po filtracji", callback=show_filtered_plot_callback)
                dpg.add_button(label="Pokaz wykres częstotliwości po filtracji", callback=show_filtered_spectrum_callback)

                dpg.add_button(label="Pokaz wykres amplitudy orginalnego pliku", callback=show_original_plot_callback)
                dpg.add_button(label="Pokaz wykres częstotliwości orginalnego pliku", callback=show_original_spectrum_callback)

            with dpg.collapsing_header(label="SNR", default_open=True):
                dpg.add_button(label="Oblicz SNR", callback=show_snr_analysis)

            with dpg.collapsing_header(label="Audio", default_open=True):
                dpg.add_button(label="Audio playback", callback=call_create_audio_play_callback)
                dpg.add_button(label="Audio filter playback", callback=call_create_audio_filtered_play_callback)

            with dpg.collapsing_header(label="Ustawienia", default_open=False):
                dpg.add_button(label="Zapisz layout", tag="save_layout", callback=save_window_config_callback)
                dpg.add_button(label="Generuj sygnał sin", callback=open_gen_signal_window)
                dpg.add_button(label="Zaszum plik", callback=open_add_noise_window_callback)
                     

dpg.setup_dearpygui()
dpg.show_viewport()

# Custom main loop
while dpg.is_dearpygui_running():
    dpg.render_dearpygui_frame()

# save_window_config()
dpg.destroy_context()