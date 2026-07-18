import tkinter as tk
from tkinter import messagebox
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pyaudio
import time
from functools import partial

class DSPTrainerApp:
    """
    A GUI application for demonstrating basic DSP concepts (filtering, FFT).
    """
    def __init__(self, master):
        self.master = master
        master.title("Digital Signal Processing Trainer")

        # --- Global Audio State ---
        self.audio_data = None # Raw recorded signal
        self.noisy_data = None # Signal after noise addition
        self.filtered_data = None
        self.sample_rate = 44100  # Standard sample rate
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.recording = False
        self.saved_tracks = {}
        self.track_counter = 0

        # --- GUI Setup ---
        self.setup_gui()

    def setup_gui(self):
        """Sets up the main layout using frames."""
        main_frame = tk.Frame(self.master, padx=10, pady=10)
        main_frame.pack(fill='both', expand=True)

        # 1. Control Panel (Top)
        control_frame = tk.Frame(main_frame, pady=10, padx=5)
        control_frame.pack(fill='x')
        
        tk.Label(control_frame, text="Recording Status:").pack(side=tk.LEFT, padx=5)
        self.record_button = tk.Button(control_frame, text="🎙 Record Audio", command=self.toggle_recording)
        self.record_button.pack(side=tk.LEFT, padx=10)

        noise_frame = tk.Frame(control_frame)
        noise_frame.pack(side=tk.LEFT, padx=(0, 20))

        tk.Label(noise_frame, text="Noise Magnitude:").pack(side=tk.LEFT, padx=(0, 5))
        self.noise_slider = tk.Scale(
            noise_frame, 
            from_=0.0, 
            to=0.5, 
            orient=tk.HORIZONTAL, 
            command=self.update_noise_label, # Update label on slide
            resolution=0.01,
            length=200
        )
        self.noise_slider.set(0.0)
        self.noise_slider.pack(side=tk.LEFT, padx=(0, 10))
        
        self.noise_label = tk.Label(noise_frame, text="0.0")
        self.noise_label.pack(side=tk.LEFT)

        # 2. Filter Editor (Middle Left)
        filter_frame = tk.LabelFrame(main_frame, text="Filter Editor", padx=10, pady=10)
        filter_frame.pack(side=tk.LEFT, padx=10, expand=True, fill="y")

        tk.Label(filter_frame, text="Filter Type:").pack(pady=5)
        self.filter_type = tk.StringVar(value="Low-Pass")
        self.filter_options = ["Low-Pass", "High-Pass", "Band-Pass", "Band-Stop"]
        tk.OptionMenu(filter_frame, self.filter_type, *self.filter_options).pack(pady=5, anchor="w")
        
        # Filter parameters
        tk.Label(filter_frame, text="Cutoff/Center Freq (Hz):").pack(pady=2)
        self.cutoff_var = tk.DoubleVar(value=3000)
        tk.Entry(filter_frame, textvariable=self.cutoff_var, width=10).pack(pady=2)

        tk.Label(filter_frame, text="Bandwidth (Hz):").pack(pady=2)
        self.bw_var = tk.DoubleVar(value=100)
        tk.Entry(filter_frame, textvariable=self.bw_var, width=10).pack(pady=2)
        
        tk.Button(filter_frame, text="⚙ Update Audio", command=self.process_audio, bg='#4CAF50', fg='white').pack(pady=15, fill='x')

        # 3. Visualization and Playback (Middle Right)
        right_frame = tk.LabelFrame(main_frame, padx=10, pady=10)
        right_frame.pack(side=tk.RIGHT, padx=10, expand=True, fill="both")
        viz_frame = tk.LabelFrame(right_frame, text="Visualization & Output", padx=10, pady=10)
        viz_frame.pack(side=tk.TOP, padx=10, expand=True, fill="both")

        # Matplotlib Canvas for plots
        self.fig, self.axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill='both', expand=True)

        # Controls below plots
        output_controls = tk.Frame(viz_frame, pady=10)
        output_controls.pack(fill='x')
        
        self.play_button_raw = tk.Button(output_controls, text="🔊 Play Raw Audio", command=partial(self.playback_audio, 'raw'), state=tk.DISABLED)
        self.play_button_raw.pack(side=tk.LEFT, padx=10)
        self.play_button_noisy = tk.Button(output_controls, text="🔊 Play Noisy Audio", command=partial(self.playback_audio, 'noisy'), state=tk.DISABLED)
        self.play_button_noisy.pack(side=tk.LEFT, padx=10)
        self.play_button_processed = tk.Button(output_controls, text="🔊 Play Processed Audio", command=partial(self.playback_audio, 'processed'), state=tk.DISABLED)
        self.play_button_processed.pack(side=tk.LEFT, padx=10)
        
        tk.Button(output_controls, text="Clear Data", command=self.clear_data).pack(side=tk.LEFT, padx=10)
        
        # save audio tracks
        saved_frame = tk.LabelFrame(right_frame, text="Saved Tracks", padx=10, pady=10)
        saved_frame.pack(side=tk.BOTTOM, padx=10, expand=True, fill="y")

        tk.Button(saved_frame, text="💾 Save Filtered Track", command=self.save_track, bg='#FFC107', fg='black').pack(side=tk.TOP, pady=5, fill='x')

        tk.Label(saved_frame, text="Track Name:").pack(side=tk.LEFT, padx=(30, 5))
        self.new_track_name = tk.StringVar(value="Untitled Sample")
        tk.Entry(saved_frame, textvariable=self.new_track_name, width=16).pack(side=tk.LEFT, padx=(5, 15))

        self.track_buttons_frame = tk.Frame(saved_frame, padx=5, pady=5, bd=2, relief=tk.GROOVE)
        self.track_buttons_frame.pack(fill='x', side=tk.BOTTOM, expand=True)
        
        # Initial update of the saved tracks UI
        self._update_saved_tracks_ui()
        
        # Initialize the noise label
        self.update_noise_label(self.noise_slider.get())


    # ========================================================
    # NOISE HANDLERS
    # ========================================================

    def update_noise_label(self, value):
        """Updates the label displaying the current noise magnitude."""
        self.noise_label.config(text=f"{float(value):.2f}")

    def add_noise(self, signal, noise_magnitude):
        """Adds Gaussian noise to the original signal."""
        if noise_magnitude <= 0.0:
            return signal.copy()
        
        # Generate noise with zero mean and specified standard deviation
        noise = np.random.normal(0, noise_magnitude, len(signal))
        
        # Add noise to the signal
        noisy_signal = signal * (1 + noise)
        return noisy_signal

    # ========================================================
    # CORE DSP FUNCTIONS
    # ========================================================

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for PyAudio stream."""
        if self.recording:
            # Convert raw bytes to numpy array and append
            np_data = np.frombuffer(in_data, dtype=np.int16)
            self.frames.append(np_data)
            return (None, pyaudio.paContinue)
        else:
            return (None, pyaudio.paContinue)

    def toggle_recording(self):
        """Starts or stops the recording."""
        if self.recording:
            self.recording = False
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
            # Process the gathered frames
            if self.frames:
                # Concatenate all recorded frames into one numpy array
                self.audio_data = np.concatenate(self.frames).astype(np.float32) / 32768.0
                self.frames = []
                messagebox.showinfo("Recording Complete", f"Recorded {len(self.audio_data)} samples.")
                self.record_button.config(text="🎙 Record Audio")
                
                # Immediately add noise and update visualizations
                self.process_audio()

                # Update UI
                self.play_button_raw.config(state=tk.NORMAL)
                self.play_button_noisy.config(state=tk.NORMAL)
            else:
                messagebox.showwarning("No Data", "No audio recorded.")
                self.clear_data()

        else:
            # Start recording
            self.recording = True
            self.frames = []
            self.audio_data = np.zeros(1)
            self.noisy_data = None # Reset noise data
            self.filtered_data = None
            self.master.update()
            self.record_button.config(text="🔴 RECORDING...")
            self.stream = self.p.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=self.sample_rate,
                                      input=True,
                                      frames_per_buffer=2048,
                                      stream_callback=self._audio_callback)

    def save_track(self):
        """Saves the currently filtered audio data and updates the UI."""
        if self.filtered_data is None:
            messagebox.showwarning("Save Error", "Please apply a filter first to generate a track to save.")
            return

        self.track_counter += 1
        track_id = self.new_track_name.get()
        
        # Store the data
        self.saved_tracks[track_id] = self.filtered_data
        
        # Update the UI
        self._create_saved_track_button(track_id, f"Saved Track {self.track_counter}")
        self._update_saved_tracks_ui()
        
        messagebox.showinfo("Success", f"Track saved successfully as {track_id}!")

    def _create_saved_track_button(self, track_id, text):
        """Internal function to create and place a button for a saved track."""
        button = tk.Button(self.track_buttons_frame, text=text, command=lambda: self.play_saved_track(track_id), padx=10, pady=5)
        button.pack(side=tk.LEFT, padx=5)

    def _update_saved_tracks_ui(self):
        """Clears and repopulates the saved tracks container with buttons."""
        # Clear all existing widgets in the track frame
        for widget in self.track_buttons_frame.winfo_children():
            widget.destroy()

        if not self.saved_tracks:
            tk.Label(self.track_buttons_frame, text="No tracks saved yet.", fg='gray').pack(padx=10, pady=10)
            return

        # Recreate buttons for all saved tracks
        for track_id, _ in self.saved_tracks.items():
            button_text = f"▶️ {track_id}"
            self._create_saved_track_button(track_id, button_text)

    def play_saved_track(self, track_id):
        """Retrieves data from the saved dictionary and plays it."""
        if track_id not in self.saved_tracks:
            messagebox.showerror("Error", "Track not found.")
            return
        
        audio_output = self.saved_tracks[track_id]
        
        # Convert float data back to 16-bit integer format
        audio_output_int = (audio_output * 32767).astype(np.int16)
        
        try:
            self.stream = self.p.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=self.sample_rate,
                                      output=True)
            self.stream.write(audio_output_int.tobytes())
            self.stream.close()
            self.stream = None
        except Exception as e:
            messagebox.showerror("Playback Error", f"Could not play saved audio: {e}")

    def clear_data(self):
        """Clears all stored audio data and plots."""
        self.audio_data = None
        self.filtered_data = None
        self.play_button_raw.config(state=tk.DISABLED)
        self.play_button_processed.config(state=tk.DISABLED)
        self.axes[0].clear()
        self.axes[1].clear()
        self.canvas.draw()
        print("Data cleared.")

    # ========================================================
    # DSP CORE LOGIC
    # ========================================================

    def apply_filter(self, data, filter_type, cutoff, bw):
        """
        Applies the selected digital filter to the audio data.
        Uses the Butterworth filter design for simplicity.
        """
        print(f"\n--- Applying {filter_type} Filter ---")
        
        # Determine filter order (a standard value)
        N = 5
        
        # Determine normalized frequency (or frequencies)
        if filter_type == "Band-Pass" or filter_type == "Band-Stop":
            # Ensure cutoff - bw > 0 and cutoff + bw < sample_rate/2
            if (cutoff - bw) < 1 or (cutoff + bw) > (self.sample_rate / 2) - 1:
                 print("Warning: Band parameters outside valid frequency range. Using default.")
                 return data.copy()
            
            cutoff_low = (cutoff - bw/2) / (self.sample_rate / 2)
            cutoff_high = (cutoff + bw/2) / (self.sample_rate / 2)
            Wn = [cutoff_low, cutoff_high]
        else:
            Wn = cutoff / (self.sample_rate / 2)

        x = data.copy()
        
        if filter_type == "Low-Pass":
            b, a = signal.butter(N, Wn, btype='low')
        
        elif filter_type == "High-Pass":
            b, a = signal.butter(N, Wn, btype='high')
        
        elif filter_type == "Band-Pass":
            b, a = signal.butter(N, Wn, btype='band')

        elif filter_type == "Band-Stop":
            b, a = signal.butter(N, Wn, btype='bandstop')

        y = signal.filtfilt(b, a, x)
        
        return y

    def _normalize_audio(self, data):
        """Scales the audio data so that its maximum absolute amplitude is 1.0."""
        if data is None or len(data) == 0:
            return data

        # Calculate the peak amplitude (the maximum absolute value)
        max_amp = np.max(np.abs(data))
        
        if max_amp == 0:
            return data
        
        # Normalize by dividing the entire signal by the maximum amplitude
        normalized_data = data / max_amp
        return normalized_data

    def process_audio(self):
        """Main function to orchestrate noise addition, filtering, and visualization."""
        if self.audio_data is None:
            messagebox.showerror("Error", "Please record audio first.")
            return

        try:
            # 1. Add Noise
            noise_mag = float(self.noise_slider.get())
            self.noisy_data = self.add_noise(self.audio_data, noise_mag)
            self.normalized_noisy_data = self._normalize_audio(self.noisy_data)

            # 2. Get Filter Parameters
            filter_type = self.filter_type.get()
            cutoff = self.cutoff_var.get()
            bw = self.bw_var.get()

            if cutoff <= 0:
                 messagebox.showwarning("Input Error", "Cutoff frequency must be positive.")
                 return
            
            # 3. Apply Filter to Noisy Data
            filtered_data = self.apply_filter(self.normalized_noisy_data, filter_type, cutoff, bw)
            self.filtered_data = filtered_data
            
            #messagebox.showinfo("Success", "Processing applied successfully! Results displayed.")
            
            # 4. Update UI
            self.update_visualization()
            self.play_button_processed.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred during processing: {e}")
            self.filtered_data = None

    # ========================================================
    # VISUALIZATION & PLAYBACK (Requires updating which data is plotted)
    # ========================================================

    def update_visualization(self):
        """Updates the waveform and FFT plots."""
        
        # --- Waveform Plotting ---
        self.axes[0].clear()
        time_axis = np.arange(len(self.audio_data)) / self.sample_rate
        
        # Plot Original Waveform
        self.axes[0].plot(time_axis, self.audio_data, label='Original Signal', color='blue', alpha=0.5)
        
        # Plot Noisy Waveform
        if self.noisy_data is not None:
            self.axes[0].plot(time_axis, self.noisy_data, label='Noisy Signal', color='orange', alpha=0.7)
            
        # Plot Filtered Waveform
        if self.filtered_data is not None:
            self.axes[0].plot(time_axis, self.filtered_data, label='Processed Signal', color='red', alpha=0.7)
            self.axes[0].legend()

        self.axes[0].set_title("Time Domain Waveform (Raw -> Noisy -> Filtered)")
        self.axes[0].set_xlabel("Time (seconds)")
        self.axes[0].set_ylabel("Amplitude")
        self.axes[0].grid(True)

        # --- FFT Plotting ---
        self.axes[1].clear()
        N = len(self.audio_data)
        
        # Calculate FFT for Original
        yf_original = np.fft.fft(self.audio_data)
        xf_original = np.fft.fftfreq(N, 1/self.sample_rate)
        magnitude_original = 2.0/N * np.abs(yf_original)
        
        # Calculate FFT for Noisy (if noise was added)
        magnitude_noisy = None
        if self.noisy_data is not None:
            yf_noisy = np.fft.fft(self.noisy_data)
            magnitude_noisy = 2.0/N * np.abs(yf_noisy)
            
            self.axes[1].plot(xf_original[:N//2], magnitude_original[:N//2], label='Original Spectrum', alpha=0.5, color='blue')
            self.axes[1].plot(xf_original[:N//2], magnitude_noisy[:N//2], label='Noisy Spectrum', alpha=0.7, color='orange')
        else:
            self.axes[1].plot(xf_original[:N//2], magnitude_original[:N//2], label='Original Spectrum', color='blue')
        
        # Calculate FFT for filtered data
        if self.filtered_data is not None:
            yf_filtered = np.fft.fft(self.filtered_data)
            xf_filtered = np.fft.fftfreq(N, 1/self.sample_rate)
            magnitude_filtered = 2.0/N * np.abs(yf_filtered)

            # Plot Spectra
            self.axes[1].plot(xf_filtered[:N//2], magnitude_filtered[:N//2], label='Processed Spectrum', alpha=0.7, color='red')
            self.axes[1].legend()
        else:
            self.axes[1].legend() # Just redraw legend if nothing is plotted

        self.axes[1].set_title("Frequency Domain (FFT) - Original -> Noisy -> Filtered")
        self.axes[1].set_xlabel("Frequency (Hz)")
        self.axes[1].set_ylabel("Magnitude")
        self.axes[1].grid(True)

        # Redraw the canvas to display updates
        self.canvas.draw()

    def playback_audio(self, type: str):
        """Plays the audio data (raw, noisy, or processed)."""
        if type == 'raw':
            data = self.audio_data
        elif type == 'processed':
            data = self.filtered_data
        elif type == 'noisy': # Added for direct noisy playback
            data = self.noisy_data
        else:
            messagebox.showwarning("Playback Error", "Invalid data type")
            return

        if data is None or len(data) == 0:
            messagebox.showwarning("Playback Error", "No processed data to play.")
            return

        # Convert float data back to 16-bit integer format
        audio_output = (data * 32767).astype(np.int16)
        
        try:
            self.stream = self.p.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=self.sample_rate,
                                      output=True)
            self.stream.write(audio_output.tobytes())
            self.stream.close()
            self.stream = None
        except Exception as e:
            messagebox.showerror("Playback Error", f"Could not play audio: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DSPTrainerApp(root)
    root.mainloop()
