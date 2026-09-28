import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PARAMETERS
# ============================================================
N_bits = 20                 # Number of bits
Tb = 1e-3                  # Bit duration
Fs = 100000                # Sampling frequency
fc = 10000                 # ASK carrier frequency
f1 = 10000                 # BFSK frequency for bit 1
f0 = 14000                 # BFSK frequency for bit 0
SNR_dB = 8                 # Channel SNR

Ns = int(Fs * Tb)          # Samples per bit
t_bit = np.arange(Ns) / Fs

# Random binary data
bits = np.random.randint(0, 2, N_bits)

# ============================================================
# CHANNEL
# ============================================================
def awgn_channel(signal, snr_db):
    """Add AWGN noise to a signal."""
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise


# ============================================================
# ASK TRANSMITTER
# ============================================================
def ask_transmitter(bits, fc, Tb, Fs):
    """Binary ASK transmitter."""
    Ns = int(Fs * Tb)
    t = np.arange(Ns) / Fs

    carrier = np.cos(2 * np.pi * fc * t)

    signal = []

    for bit in bits:
        # Bit 1 -> carrier transmitted
        # Bit 0 -> no carrier
        amplitude = 1 if bit == 1 else 0
        signal.extend(amplitude * carrier)

    return np.array(signal)


# ============================================================
# ASK COHERENT RECEIVER
# ============================================================
def ask_coherent_receiver(rx, fc, Tb, Fs):
    """Coherent ASK detector using correlation."""
    Ns = int(Fs * Tb)
    t = np.arange(Ns) / Fs

    reference = np.cos(2 * np.pi * fc * t)

    received_bits = []

    for i in range(0, len(rx), Ns):
        symbol = rx[i:i + Ns]

        if len(symbol) < Ns:
            break

        # Correlation with carrier
        correlation = np.sum(symbol * reference)

        # Threshold
        if correlation > 0:
            received_bits.append(1)
        else:
            received_bits.append(0)

    return np.array(received_bits)


# ============================================================
# BFSK TRANSMITTER
# ============================================================
def bfsk_transmitter(bits, f0, f1, Tb, Fs):
    """Binary FSK transmitter."""
    Ns = int(Fs * Tb)
    t = np.arange(Ns) / Fs

    signal = []

    for bit in bits:
        if bit == 1:
            tone = np.cos(2 * np.pi * f1 * t)
        else:
            tone = np.cos(2 * np.pi * f0 * t)

        signal.extend(tone)

    return np.array(signal)


# ============================================================
# BFSK NONCOHERENT RECEIVER
# ============================================================
def bfsk_noncoherent_receiver(rx, f0, f1, Tb, Fs):
    """
    Noncoherent BFSK detector.
    Uses energy detection at the two possible frequencies.
    """
    Ns = int(Fs * Tb)
    t = np.arange(Ns) / Fs

    ref0_cos = np.cos(2 * np.pi * f0 * t)
    ref0_sin = np.sin(2 * np.pi * f0 * t)

    ref1_cos = np.cos(2 * np.pi * f1 * t)
    ref1_sin = np.sin(2 * np.pi * f1 * t)

    received_bits = []

    for i in range(0, len(rx), Ns):
        symbol = rx[i:i + Ns]

        if len(symbol) < Ns:
            break

        # Energy at f0
        I0 = np.sum(symbol * ref0_cos)
        Q0 = np.sum(symbol * ref0_sin)
        E0 = I0**2 + Q0**2

        # Energy at f1
        I1 = np.sum(symbol * ref1_cos)
        Q1 = np.sum(symbol * ref1_sin)
        E1 = I1**2 + Q1**2

        # Choose frequency with greater energy
        if E1 > E0:
            received_bits.append(1)
        else:
            received_bits.append(0)

    return np.array(received_bits)


# ============================================================
# TRANSMISSION
# ============================================================

# ASK
ask_tx = ask_transmitter(bits, fc, Tb, Fs)
ask_rx = awgn_channel(ask_tx, SNR_dB)

# BFSK
bfsk_tx = bfsk_transmitter(bits, f0, f1, Tb, Fs)
bfsk_rx = awgn_channel(bfsk_tx, SNR_dB)


# ============================================================
# RECEIVERS
# ============================================================

ask_bits = ask_coherent_receiver(
    ask_rx, fc, Tb, Fs
)

bfsk_bits = bfsk_noncoherent_receiver(
    bfsk_rx, f0, f1, Tb, Fs
)


# ============================================================
# BER
# ============================================================

ask_errors = np.sum(bits != ask_bits)
bfsk_errors = np.sum(bits != bfsk_bits)

ask_ber = ask_errors / N_bits
bfsk_ber = bfsk_errors / N_bits

print("Original bits:")
print(bits)

print("\nASK received bits:")
print(ask_bits)

print("\nBFSK received bits:")
print(bfsk_bits)

print("\nASK BER =", ask_ber)
print("BFSK BER =", bfsk_ber)


# ============================================================
# PLOTS
# ============================================================

time = np.arange(len(bits) * Ns) / Fs

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.step(np.arange(N_bits) * Tb,
         bits,
         where='post')
plt.title("Input Binary Data")
plt.ylabel("Bit")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(time, ask_tx)
plt.title("ASK Transmitted Signal")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(time, bfsk_tx)
plt.title("BFSK Transmitted Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()
