'''
This script parses Psi4 log files to save multiple properties such as geometries, thermodynamic quantities, frequencies, normal modes etc. into numpy arrays
'''

import numpy as np
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import time
import sys

def print_write(string, filename):
    """
    Print the string to console and append it to the specified file.
    """
    print(string)
    outfile = open(filename, 'a')
    outfile.write(string)
    outfile.close()

def parse_freq(line):
    """
    Parse vibrational frequencies from a reversed output file.
    Frequencies are complex numbers. Imaginary frequencies are represented using 'j'.
    Only frequencies listed under 'Freq [cm^-1]' are extracted.
    """
    file = open(path+line[:-1], 'r')
    out = file.readlines()
    file.close()
    out = list(reversed(out))

    try:
        tmp = []
        for lin in out:
            if 'Freq [cm^-1]' in lin:
                for num in lin.split()[2:]:
                    if 'i' not in num:
                        tmp.append(np.complex_(num))
                    else:
                        tmp.append(np.complex_(num[:-1]+'j'))  # Imaginary frequency
            if 'post-proj  all modes' in lin:
                break
    except AttributeError:
        print('Could not read the contents of output file ' + str(path+line[:-1]) + '.')
    except UnboundLocalError:
        print('Some or all vibrational frequencies could not be found for ' + str(path+line[:-1]) + '.')

    # Arrange extracted frequencies into an array
    freqs = np.zeros(len(tmp), dtype='complex_')
    num_sets = len(tmp) // 3
    i = 0
    if num_sets > 0:
        while (i / 3) != num_sets:
            if i == 0:
                freqs[i:i+3] = tmp[-(i+3):]
            else:
                freqs[i:i+3] = tmp[-(i+3):-i]
            i += 3
        freqs[i:] = tmp[:-i]

    return line[:-1], freqs

def parse_vibrational_modes(line):
    """
    Extract vibrational mode displacements from the output file.
    Returns modes grouped by frequency triplets. Each mode is a vector of 3D displacements.
    """
    file = open(path+line[:-1], 'r')
    out = file.readlines()
    file.close()
    out = list(reversed(out))

    try:
        v_modes = []
        for i in range(len(out)):
            if 'Freq [cm^-1]' in out[i]:
                tmp = out[i].split()
                while tmp[0] != '1':  # Look for line where displacements start
                    i = i-1
                    tmp = out[i].split()
                freq1_modes, freq2_modes, freq3_modes = [], [], []
                while out[i].split() != []:
                    split = out[i].split()
                    split = list(np.float_(split[2:]))
                    freq1_modes.append(split[0:3])
                    freq2_modes.append(split[3:6])
                    freq3_modes.append(split[6:9])
                    i = i-1
                if freq2_modes != [[], []]:
                    v_modes.append(freq1_modes)
                    v_modes.append(freq2_modes)
                    v_modes.append(freq3_modes)
                else:
                    v_modes.append(freq1_modes)

            if 'post-proj  all modes' in out[i]:
                break
    except AttributeError:
        print('Could not read the contents of output file ' + str(path+line[:-1]) + '.')
    except UnboundLocalError:
        print('Some or all vibrational modes could not be found for ' + str(path+line[:-1]) + '.')

    v_modes = np.array(v_modes)
    modes_shape = v_modes.shape
    if len(modes_shape) > 2:
        # Rearranging order so modes are chronological, not reversed
        modes = np.zeros((modes_shape[0], modes_shape[1], modes_shape[2]))
        num_sets = modes_shape[0] // 3
        i = 0
        if num_sets > 0:
            while (i / 3) != num_sets:
                if i == 0:
                    modes[i:i+3:] = v_modes[-(i+3)::]
                else:
                    modes[i:i+3:] = v_modes[-(i+3):-i:]
                i += 3
            modes[i::] = v_modes[:-i:]

        return modes
    else:
        return v_modes

def parse_zpves(line):
    """
    Parse zero-point vibrational energy (ZPVE) from the output file.
    """
    file = open(path+line[:-1], 'r')
    out = file.readlines()
    file.close()
    out = list(reversed(out))

    try:
        for lin in out:
            if 'Vibrational ZPE' in lin:
                split = lin.split()
                zpve = float(split[6])
                break
        return zpve
    except AttributeError:
        print('Could not read the contents of output file ' + str(path+line[:-1]) + '.')
    except UnboundLocalError:
        print('A zero-point vibrational frequency could not be found for ' + str(path+line[:-1]) + '.')

def parse_internal_energy_0K(line):
    """
    Parse total internal energy at 0 K (includes ZPE) from the output file.
    """
    file = open(path+line[:-1], 'r')
    out = file.readlines()
    file.close()
    out = list(reversed(out))

    try:
        for lin in out:
            if 'Total ZPE' in lin:
                split = lin.split()
                internal_energy_0K = float(split[7])
                break
        return internal_energy_0K
    except AttributeError:
        print('Could not read the contents of output file ' + str(path+line[:-1]) + '.')
    except UnboundLocalError:
        print('Some or all vibrational frequencies could not be found for ' + str(path+line[:-1]) + '.')

def parse_internal_energy(line):
    """
    Parse internal energy at 298.15 K from the output file.
    """
    file = open(path+line[:-1], 'r')
    out = file.readlines()
    file.close()
    out = list(reversed(out))

    try:
        for lin in out:
            if 'Total E, Electronic energy at  298.15 [K]' in lin:
                split = lin.split()
                internal_energy = float(split[7])
                break
        return internal_energy
    except AttributeError:
        print('Could not read the contents of output file ' + str(path+line[:-1]) + '.')
    except UnboundLocalError:
        print('An internal energy could not be found for ' + str(path+line[:-1]) + '.')
    except ValueError:
        print(line)

def parse_enthalpy(line):
    """
    Parse enthalpy at 298.15 K from the output file.
    """
    file = open(path+line[:-1], 'r')
    out = file.readlines()
    file.close()
    out = list(reversed(out))

    try:
        for lin in out:
            if 'Total H, Enthalpy at  298.15 [K]' in lin:
                split = lin.split()
                enthalpy = float(split[6])
                break
        return enthalpy
    except AttributeError:
        print('Could not read the contents of output file ' + str(path+line[:-1]) + '.')
    except UnboundLocalError:
        print('An enthalpy could not be found for ' + str(path+line[:-1]) + '.')

def parse_entropy(line):
    """
    Parse entropy at 298.15 K from the output file.
    """
    file = open(path+line[:-1], 'r')
    out = file.readlines()
    file.close()
    out = list(reversed(out))

    try:
        for lin in out:
            if 'Total S' in lin:
                split = lin.split()
                entropy = float(split[8])
                break
        return entropy
    except AttributeError:
        print('Could not read the contents of output file ' + str(path+line[:-1]) + '.')
    except UnboundLocalError:
        print('An entropy could not be found for ' + str(path+line[:-1]) + '.')

def parse_free_energy(line):
    """
    Parse free energy (Gibbs free enthalpy) at 298.15 K from the output file.
    """
    file = open(path+line[:-1], 'r')
    out = file.readlines()
    file.close()
    out = list(reversed(out))

    try:
        for lin in out:
            if 'Free enthalpy at  298.15 [K]' in lin:
                split = lin.split()
                free_energy = float(split[7])
                break
        return free_energy
    except AttributeError:
        print('Could not read the contents of output file ' + str(path+line[:-1]) + '.')
    except UnboundLocalError:
        print('A free energy could not be found for ' + str(path+line[:-1]) + '.')

def parse_Cv(line):
    """
    Parse constant-volume heat capacity (Cv) at 298.15 K from the output file.
    """
    file = open(path+line[:-1], 'r')
    out = file.readlines()
    file.close()
    out = list(reversed(out))

    try:
        for lin in out:
            if 'Total Cv' in lin:
                split = lin.split()
                Cv = float(split[2])
                break
        return Cv
    except AttributeError:
        print('Could not read the contents of output file ' + str(path+line[:-1]) + '.')
    except UnboundLocalError:
        print('A fixed-volume heat capacity could not be found for ' + str(path+line[:-1]) + '.')

def parse_Cp(line):
    """
    Parse constant-pressure heat capacity (Cp) at 298.15 K from the output file.
    """
    file = open(path+line[:-1], 'r')
    out = file.readlines()
    file.close()
    out = list(reversed(out))

    try:
        for lin in out:
            if 'Total Cp' in lin:
                split = lin.split()
                Cp = float(split[2])
                break
        return Cp
    except AttributeError:
        print('Could not read the contents of output file ' + str(path+line[:-1]) + '.')
    except UnboundLocalError:
        print('A fixed-pressure heat capacity could not be found for ' + str(path+line[:-1]) + '.')


def time_reducer(t):
    times = []
    if 60 < t < 3600:
        minutes = t // 60
        seconds = t % 60
        times.append(seconds), times.append(minutes), times.append(0)
    elif t > 3600:
        hours = t // 3600
        minutes = (t - (hours * 3600)) // 60
        seconds = t % 60
        times.append(seconds), times.append(minutes), times.append(hours)
    else:
        times.append(t), times.append(0), times.append(0)

    return times

inputs = sys.argv[1:]
#print(inputs)

partitions = []

while True:
    valid_input = True  # Assume the input is valid until proven otherwise
    
    for element in inputs:
        try:
            num = int(element.strip())
            if 1 <= num <= 100:
                partitions.append(num)
            else:
                print('Please make sure your numbers are between 1 and 100.')
                valid_input = False  # Mark the input as invalid
                break  # Exit the loop if an invalid number is encountered
        except ValueError:
            print('Please make sure your input consists of valid integers.')
            valid_input = False  # Mark the input as invalid
            break  # Exit the loop if a non-integer input is encountered
    
    if valid_input:
        print('Valid partitions:', partitions)
        break  # Exit the loop if the input is valid

# Start timer
start_time = time.time()

# Iterate over data partitions
for partition in partitions:
    # Read file containing paths to Psi4 output files
    file = open(f'converged{partition}.txt', 'r')
    lines = file.readlines()
    file.close()
    
    # Set up a log file for output messages
    log = f'/home/scott/Work/qm24/qm24_psi4_{partition}_log.txt'

    # Log the number of files to be processed
    print_write('Partition ' + str(partition) + ' (' + str(len(lines)) + ' Psi4 Output Files):\n', log)
    
    # Initialize containers
    paths, freqs, vib_modes = [], [], []
    saddle_points, transition_states, higher_order_saddles = [], [], []

    # Parse frequencies and paths in parallel
    first = Parallel(n_jobs=16)(delayed(parse_freq)(line) for line in tqdm(lines))
    for i in range(len(first)):
        paths.append(first[i][0])           # File path
        freqs.append(first[i][1])           # Frequencies

    # Parse vibrational modes
    vmodes = Parallel(n_jobs=16)(delayed(parse_vibrational_modes)(line) for line in tqdm(lines))
    for i in range(len(vmodes)):
        vib_modes.append(vmodes[i])

    # Parse thermodynamic quantities
    zpves = Parallel(n_jobs=16)(delayed(parse_zpves)(line) for line in tqdm(lines))
    E0 = Parallel(n_jobs=16)(delayed(parse_internal_energy_0K)(line) for line in tqdm(lines))
    E298 = Parallel(n_jobs=16)(delayed(parse_internal_energy)(line) for line in tqdm(lines))
    H = Parallel(n_jobs=16)(delayed(parse_enthalpy)(line) for line in tqdm(lines))
    S = Parallel(n_jobs=16)(delayed(parse_entropy)(line) for line in tqdm(lines))
    G = Parallel(n_jobs=16)(delayed(parse_free_energy)(line) for line in tqdm(lines))
    Cv = Parallel(n_jobs=16)(delayed(parse_Cv)(line) for line in tqdm(lines))
    Cp = Parallel(n_jobs=16)(delayed(parse_Cp)(line) for line in tqdm(lines))

    # Log successful retrievals
    print_write('Vibrational frequencies of ' + str(len(freqs)) + ' compounds were retrieved.\n', log)
    print_write('Vibrational modes of ' + str(len(vib_modes)) + ' compounds were retrieved.\n', log)
    print_write('Zero-point vibrational energies of ' + str(len(zpves)) + ' compounds were retrieved.\n', log)
    print_write('Internal energies at 0 K of ' + str(len(E0)) + ' compounds were retrieved.\n', log)
    print_write('Internal energies at 298 K of ' + str(len(E298)) + ' compounds were retrieved.\n', log)
    print_write('Enthalpies at 298 K of ' + str(len(H)) + ' compounds were retrieved.\n', log)
    print_write('Entropies at 298 K of ' + str(len(S)) + ' compounds were retrieved.\n', log)
    print_write('Free energies at 0 K of ' + str(len(G)) + ' compounds were retrieved.\n', log)
    print_write('Fixed-volume heat capacity of ' + str(len(Cv)) + ' compounds were retrieved.\n', log)
    print_write('Fixed-pressure heat capacity of ' + str(len(Cp)) + ' compounds were retrieved.\n', log)

    # Classify saddle points based on number of imaginary frequencies
    saddles, transition_states, higher_order_saddles = [], [], []
    for i in range(len(freqs)):
        imags = []
        for num in freqs[i]:  # Check each frequency
            if num.imag != 0:
                imags.append(1)
                if len(imags) > 2:
                    break  # Early exit if more than two imaginary freqs
        
        # 1 imaginary frequency = transition state
        if len(imags) == 1:
            saddles.append(paths[i])
            transition_states.append(paths[i])
        # >1 imaginary frequency = higher-order saddle
        elif len(imags) > 1:
            saddles.append(paths[i])
            higher_order_saddles.append(paths[i])

    # Log saddle point summary
    print_write(f'{len(saddles)} saddles were found in amons{partition}, of which {len(transition_states)} are transition states and {len(higher_order_saddles)} are higher order. \n', log)

    # Save parsed data to .npz archive
    output_file = f'qm24_psi4_{partition}.npz'
    np.savez_compressed(f'/home/scott/Work/qm24/{output_file}',
                        paths=paths, freqs=freqs, vibrational_modes=vib_modes,
                        zpves=zpves, E0=E0, E298=E298, H=H, S=S, G=G, Cv=Cv, Cp=Cp,
                        dtype=object)

    # Save saddle point classification to a separate file
    saddles_output = f'amons{partition}_saddles.npz'
    np.savez_compressed(f'/home/scott/Work/qm24/{saddles_output}',
                        transition_states=transition_states,
                        higher_order_saddles=higher_order_saddles,
                        dtype=object)

    # Reload data for quick verification
    data = np.load(f'/home/scott/Work/qm24/{output_file}', allow_pickle=True)

    # Print sample outputs to verify content
    print(len(data['paths']), data['paths'][0])
    print(len(data["freqs"]), data["freqs"][0])
    print(len(data['vibrational_modes']), len(data['vibrational_modes'][0]), data['vibrational_modes'][0])
    print(len(data["zpves"]), data["zpves"][0])
    print(len(data["E0"]), data["E0"][0])
    print(len(data["E298"]), data["E298"][0])
    print(len(data["H"]), data["H"][0])
    print(len(data["G"]), data["G"][0])
    print(len(data["Cv"]), data["Cv"][0])
    print(len(data["Cp"]), data["Cp"][0])

# Stop timer
end_time = time.time()
elapsed_time = end_time - start_time

# Convert time to (seconds, minutes, hours) format
times = time_reducer(elapsed_time)

# Display final runtime
print(f'Runtime was {times[2]} hours, {times[1]} minutes, and {times[0]} seconds.')
