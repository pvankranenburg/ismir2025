import music21 as m21
import numpy as np
from scipy.optimize import minimize
from collections import defaultdict
from scipy.optimize import differential_evolution, NonlinearConstraint, dual_annealing, basinhopping
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from fractions import Fraction
import time
import sys
import copy
import math
from sympy import nsimplify

def get_semitone_interval(note1, note2):
    pitch1 = note1.pitch.midi
    pitch2 = note2.pitch.midi
    lowest_pitch = min([pitch1, pitch2])
    highest_pitch = max([pitch1, pitch2])
    raw_interval = (highest_pitch - lowest_pitch) % 12
    return raw_interval, lowest_pitch % 12, highest_pitch % 12

def analyze_intervals(score, pair_frequencies, durationWeight=True, harmonic=True, melodic=True):
    chordified = score.chordify()
    if harmonic:
        for chord in chordified.recurse().getElementsByClass('Chord'):
            notes = chord.notes
            for i in range(len(notes)):
                for j in range(i + 1, len(notes)):
                    if notes[i].isNote and notes[j].isNote:
                        interval, pitch1, pitch2 = get_semitone_interval(notes[i], notes[j])
                        #if interval in interval_types:
                        if durationWeight:
                            w_dur = chord.quarterLength
                        else:
                            w_dur = 1.0
                        pair_frequencies[(pitch1, pitch2, interval)] += chord.beatStrength * w_dur
    if melodic:
        for part in score.parts:
            notes = [n for n in part.flatten().notes if n.isNote]
            for i in range(len(notes) - 1):
                interval, pitch1, pitch2 = get_semitone_interval(notes[i], notes[i + 1])
                #if interval in interval_types:
                if durationWeight:
                    w_dur = min(notes[i].quarterLength, notes[i+1].quarterLength)
                else:
                    w_dur = 1.0
                pair_frequencies[(pitch1, pitch2, interval)] += notes[i + 1].beatStrength * w_dur

# Output contains all possible intervals.
def process_scores(score_files, harmonic=True, melodic=True):
    pair_frequencies = defaultdict(float)  # Changed to float for beatStrength
    for filepath in score_files:
        score = m21.converter.parse(filepath)
        analyze_intervals(score, pair_frequencies, harmonic=harmonic, melodic=melodic)
    return pair_frequencies

# Output contains only intervals in interval_types
# Multiplies weights by total_weight.
def compute_interval_data(pair_frequencies, interval_types, total_weight=1.0):
    # total_strength = sum(pair_frequencies.values())
    strength_selection = [pair_frequencies[key] for key in pair_frequencies.keys() if key[2] in interval_types]
    total_strength = sum(strength_selection)
    interval_data = []
    type_strengths = defaultdict(float)
    for key, strength in pair_frequencies.items():
        pitch1, pitch2, semitones = key
        if semitones in interval_types:
            type_strengths[semitones] += strength
    for key, strength in pair_frequencies.items():
        pitch1, pitch2, semitones = key
        if semitones in interval_types:
            type_weight = type_strengths[semitones] / total_strength if total_strength > 0 else 0
            weight = (strength / type_strengths[semitones]) * type_weight * total_weight
            interval_data.append([(pitch1, pitch2), interval_types[semitones]['targets'], weight])
            #print(f"Pair: {(pitch1, pitch2)}, Semitones: {semitones}, Targets: {interval_types[semitones]['targets']}, Weight: {weight}")
    return interval_data


# if p_in is a p vector incl 0 and 1200, just compute the value for the objective function
def optimize_temperament(
        interval_data,
        method='differential_evolution',
        alpha_map = {4: 10.0, 7: 10.0},
        alpha_default = 10.0,
        beta = 1.0,
        gamma = 10000.0,
        delta = 2.0,
        bh_niter = 500,
        bh_stepsize = 30.0,
        bh_T = 5.0,
        bh_stepwise_factor = 0.9,
        de_maxiter = 10000,
        de_popsize = 25,
        de_tol = 0.01, #scipy default
        de_mutation = (0.5, 1), #scipy default
        da_maxiter = 5000,
        da_initial_temp=10000,
        verbose = True,
        verbose_objective=False,
        verbose_optimizer=False,
        p_in=None,
        thirdbounds=None,
        fifthbounds=None,
        p0=np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]),
        bounds=[(75, 125), (175, 225), (275, 325), (375, 425), (475, 525), (575, 625), (675, 725), (775, 825), (875, 925), (975, 1025), (1075, 1125)]
    ):
    result = {}
    result['parameters'] = str(locals())
    all_pairs = np.array([(pair[0], pair[1]) for pair, _, _ in interval_data])
    i_indices = all_pairs[:, 0]
    j_indices = all_pairs[:, 1]
    weights = np.array([w for _, _, w in interval_data])
    targets_list = [targets for _, targets, _ in interval_data]
    max_targets = max(len(targets) for targets in targets_list)  # Max number of targets

    semitones_list = [(j - i) % 12 for i, j in all_pairs]

    # Interval-specific reward coefficients
    alpha_s = np.array([alpha_map.get(s, alpha_default) for s in semitones_list])

    # N.B. If method == 'dual_annealing', the thirdbounds and fifthbounds must be handled in the objective function
    #      For other methods, it can be in the constraints
    bounds_in_objective = True if method == 'dual_annealing' else False

    if verbose:
        print(f"{p_in=}")
        print(f"{p0=}")
        print(f"{bounds=}")
        print(f"{fifthbounds=}")
        print(f"{thirdbounds=}")
        print(f"{alpha_map=}")
        print(f"{alpha_default=}")
        print(f"{beta=}")
        print(f"{gamma=}")
        print(f"{delta=}")
        print(f"Using method: {method}")
        if method == 'basinhopping':
            print(f"    {bh_niter=}, {bh_stepsize=}, {bh_T=}, {bh_stepwise_factor=}")
        if method == 'differential_evolution':
            print(f"    {de_maxiter=}, {de_popsize=}")
        if method == 'dual_annealing':
            print(f"    {da_maxiter=}, {da_initial_temp=}")

    if verbose:
        if bounds_in_objective:
            print('Dealing with fifth and third bounds IN objective function')
        else:
            print('Dealing with fifth and third bounds IN constraints')

    def objective(p):
        p_full = np.concatenate(([0], p, [1200])) if len(p) == 11 else p
        if verbose_objective: print(f"{p_full=}")
        tempered = (p_full[j_indices % 12] - p_full[i_indices % 12]) % 1200
        if verbose_objective: print(f"{tempered=}")
        
        all_distances = np.full((len(tempered), max_targets), np.inf)
        for i, targets in enumerate(targets_list):
            all_distances[i, :len(targets)] = np.abs(tempered[i] - np.array(targets))
        
        min_indices = np.argmin(all_distances, axis=1)
        deviations_penalty = all_distances[np.arange(len(tempered)), min_indices]
        if verbose_objective: print(f"{deviations_penalty=}")
        if verbose_objective: print(f"deviations_penalty**2={deviations_penalty**2}")
        
        deviations_reward = np.where(min_indices == 0, all_distances[:, 0], np.inf)
        if verbose_objective: print(f"{deviations_reward=}")
        reward_term = alpha_s * np.exp(-beta * deviations_reward)
        reward_term[deviations_reward == np.inf] = 0
        if verbose_objective: print(f"Reward term: {reward_term.round(2)}")

        fifth_bound_penalty = 0.0
        third_bound_penalty = 0.0
        if bounds_in_objective:
            if fifthbounds is not None:
                fifth_indices = [idx for idx, s in enumerate(semitones_list) if s == 7]
                if fifth_indices:
                    fifths = np.array([(p_full[j_indices[idx] % 12] - p_full[i_indices[idx]]) % 1200 
                                    for idx in fifth_indices])
                    fifth_violations = np.maximum(fifths - fifthbounds[1], 0) + np.maximum(fifthbounds[0] - fifths, 0)
                    fifth_bound_penalty = gamma * np.sum(1.0 - np.exp(-delta * fifth_violations))
                    if verbose_objective:
                        print(f"Fifths: {fifths.round(2)}")
                        print(f"Fifth violations: {fifth_violations.round(2)}")
                        print(f"Fifth penalty: {fifth_bound_penalty}")
            if thirdbounds is not None:
                third_indices = [idx for idx, s in enumerate(semitones_list) if s == 4]
                if third_indices:
                    thirds = np.array([(p_full[j_indices[idx] % 12] - p_full[i_indices[idx]]) % 1200 
                                    for idx in third_indices])
                    third_violations = np.maximum(thirds - thirdbounds[1], 0) + np.maximum(thirdbounds[0] - thirds, 0)
                    third_bound_penalty = gamma * np.sum(1.0 - np.exp(-delta * third_violations))
                    if verbose_objective:
                        print(f"Thirds: {thirds.round(2)}")
                        print(f"Third violations: {third_violations.round(2)}")
                        print(f"Third penalty: {third_bound_penalty}")
        
        objective_value = np.sum(weights * (deviations_penalty**2 - reward_term)) + fifth_bound_penalty + third_bound_penalty
        if verbose_objective:
            print(f"Objective: {objective_value}")
        return objective_value

    def fifth_constraint(p):
        p_full = np.concatenate(([0], p, [1200])) if len(p) == 11 else p
        fifth_indices = [idx for idx, s in enumerate(semitones_list) if s == 7]
        if fifth_indices:
            fifths = np.array([(p_full[j_indices[idx] % 12] - p_full[i_indices[idx]]) % 1200 
                              for idx in fifth_indices])
            return fifths
        return np.array([])  # Return empty array if no fifths

    def third_constraint(p):
        p_full = np.concatenate(([0], p, [1200])) if len(p) == 11 else p
        third_indices = [idx for idx, s in enumerate(semitones_list) if s == 4]
        if third_indices:
            thirds = np.array([(p_full[j_indices[idx] % 12] - p_full[i_indices[idx]]) % 1200 
                              for idx in third_indices])
            return thirds
        return np.array([])  # Return empty array if no thirds

    # Constraint: All monotonicity conditions (p[i] - p[i-1] >= 10)
    def monotonicity_constraint(p):
        return [p[i] - p[i-1] - 10 for i in range(1, 11)]

    constraints = [NonlinearConstraint(monotonicity_constraint, 0, np.inf)]
    if fifthbounds != None:
        fifth_bounds_constraint = NonlinearConstraint(
            fifth_constraint, 
            [fifthbounds[0]] * len([s for s in semitones_list if s == 7]), 
            [fifthbounds[1]] * len([s for s in semitones_list if s == 7])
        )
        constraints.append(fifth_bounds_constraint)
        if verbose: print("fifth_bounds_constraint added to constraints")
    if thirdbounds != None:
        third_bounds_constraint = NonlinearConstraint(
            third_constraint, 
            [thirdbounds[0]] * len([s for s in semitones_list if s == 4]), 
            [thirdbounds[1]] * len([s for s in semitones_list if s == 4])
        )
        constraints.append(third_bounds_constraint)
        if verbose: print("third_bounds_constraint added to constraints")
    
    # Adjust bounds for thirds or fifths (on C)
    # BUT Only if the interval occurs in the data
    if thirdbounds != None:
        if (0,4) in [idat[0] for idat in interval_data]:
            bounds[3] = thirdbounds
            if verbose: print("Updated bounds for thirdbounds: ", bounds)
    if fifthbounds != None:
        if (0,7) in [idat[0] for idat in interval_data]:
            bounds[6] = fifthbounds
            if verbose: print("Updated bounds for fifthbounds: ", bounds)

    def check_constraints(p_cand):
        if len(p_cand) == 13:
            p_cand = p_cand[1:12]
        bounds_violated = False
        for i, (lower, upper) in enumerate(bounds):
            if p_cand[i] < lower or p_cand[i] > upper:
                if verbose: print(f"Element {i} = {p_cand[i]} violates bounds ({lower}, {upper})")
                bounds_violated = True
        constraints_violated = False
        for j, nlc in enumerate(constraints):
            constraint_values = nlc.fun(p_cand)
            constraint_values = np.atleast_1d(constraint_values)
            for i, (val, lb, ub) in enumerate(zip(constraint_values, np.atleast_1d(nlc.lb), np.atleast_1d(nlc.ub))):
                if val < lb or val > ub:
                    if verbose: print(f"Constraint {j}, component {i} violated: {lb} <= {val} <= {ub}")
                    constraints_violated = True
        return bounds_violated, constraints_violated

    if p_in is not None:
        bounds_violated, constraints_violated = check_constraints(p_in)
        if bounds_violated or constraints_violated:
            if verbose_objective:
                objective(p_in[1:12]) # do compute objective in case of bound violation, but verbose
            result['obj_val'] = 'inf'
            result['tuning'] = p_in
            result['result'] = ''
            return result
        else:
            result['obj_val'] = objective(p_in[1:12])
            result['tuning'] = p_in
            result['result'] = ''
            return result

    p0 = p0[1:12] if len(p0) == 13 else p0

    if method == 'dual_annealing':
        result_optim = dual_annealing(
            objective,
            bounds,
            maxiter=da_maxiter,
            x0=p0,
            initial_temp=da_initial_temp,
        )
        tuning = np.concatenate(([0], result_optim.x, [1200]))
        if verbose and verbose_optimizer: print("dual_annealing does not support verbose feedback at runtime.")
        if verbose: print("Optimized Tuning (cents):", np.array2string(tuning.round(2), separator=','), 'Result: ', result_optim)
        if verbose: print("Objective function value: ", result_optim.fun)

    if method == 'differential_evolution':
        #if verbose: print("Using strategy: rand2bin")
        result_optim = differential_evolution(
            objective,
            bounds,
            tol=de_tol,
            mutation=de_mutation,
            maxiter=de_maxiter,
            popsize=de_popsize, 
            constraints=constraints,
            polish=True,
            x0=p0,
            disp=verbose_optimizer
        )
        tuning = np.concatenate(([0], result_optim.x, [1200]))
        if verbose: print("Optimized Tuning (cents):", np.array2string(tuning.round(2), separator=','), 'Result: ', result_optim)
        if verbose: print("Objective function value: ", result_optim.fun)

    if method == 'basinhopping':
        result_optim = basinhopping(
            objective,
            x0=p0,
            niter=bh_niter,
            minimizer_kwargs={"method": "COBYLA", "bounds": bounds, "constraints": constraints},
            stepsize=bh_stepsize,
            stepwise_factor=bh_stepwise_factor,
            T=bh_T,
            disp=verbose_optimizer,
        )
        tuning = np.concatenate(([0], result_optim.x, [1200]))
        if verbose: print("Basin-Hopping Status:", result_optim.lowest_optimization_result.success, result_optim.lowest_optimization_result.message)
        if verbose: print("Optimized Tuning (cents):", np.array2string(tuning.round(2), separator=','), 'Result: ', result_optim)
        if verbose: print("Objective function value: ", result_optim.fun)

    # check final tuning:
    if verbose: print("Checking bounds and constraints")
    bounds_violated, constraints_violated = check_constraints(tuning)
    obj_val = np.inf if bounds_violated or constraints_violated else result_optim.fun
    result['obj_val'] = obj_val
    result['tuning'] = tuning
    result['result'] = str(result_optim)
    return result

# p_vector includes 0 and 1200
def write_to_scl_file(p_vector, filename, description="Well-Tempered Clavier Tuning"):
    """
    Writes a pitch vector (in cents) to a Scala .scl file.
    
    Parameters:
    - p_vector: List or array of pitch values in cents (e.g., [0.0, 97.9, ..., 1200.0]).
    - filename: String, output file name (e.g., 'wtc_tuning.scl').
    - description: String, description of the tuning (default: "Well-Tempered Clavier Tuning").
    """
    # Ensure filename ends with .scl
    if not filename.endswith('.scl'):
        filename += '.scl'
    
    # Number of notes is length of p_vector minus 1 (excluding octave if listed)
    num_notes = len(p_vector) - 1  # 12 notes + octave
    
    # Open file for writing
    with open(filename, 'w') as f:
        # Write header
        f.write(f"! {filename}\n")
        f.write(f"{description}\n")
        f.write(f"{num_notes}\n")
        
        # Write pitch values (excluding the initial 0.0, as Scala assumes root is 0)
        for pitch in p_vector[1:]:  # Start from second value (after 0.0)
            f.write(f"{pitch:.6f}\n")  # 6 decimal places for precision

def cents2ratio(cents):
    return 2 ** (cents/1200.)

def ratio2cents(ratio: Fraction):
    return 1200 * math.log2(ratio) / math.log2(2.0)

def getTimidityFrequencies(p_vector, f0=440, midi0=69):
    tun = np.array(p_vector[:12])
    tun_refA = tun - tun[9]
    tun_baseA = np.concatenate((tun_refA[9:], tun_refA[:9]+1200.))
    ratios = [cents2ratio(c) for c in tun_baseA]
    freqs = []
    for i in range(128):
        (octave, index)=divmod( i-midi0, len(ratios) )
        freqs.append( int( 2**octave*ratios[index]*f0*1000+0.5) )
    return freqs

def writeTimidityFrequencies(p_vector, fname, f0=440, midi0=69):
    freqs = getTimidityFrequencies(p_vector, f0=f0, midi0=midi0)
    with open(fname, 'w') as f:
        for freq in freqs:
            f.write(str(freq))
            f.write('\n')

def computeIntervalSizes(tuning_in):
    tuning = {x: tuning_in[x] for x in range(12)}
    sizes = np.zeros((12,12))
    for pitch in range(12):
        for interval in range(12):
            sizes[pitch,interval] = (tuning[(pitch+interval)%12] - tuning[pitch]) % 1200
    return sizes

def intervalSizes_str(intervalsizes):
    result = ''
    pitches = { 0: "C", 1: "C#", 2: "D", 3: "Eb", 4: "E", 5: "F", 6: "F#", 7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B" }
    intervals = { 0: "P1", 1: "m2", 2: "M2", 3: "m3", 4: "M3", 5: "P4", 6: "A4-d5", 7: "P5", 8: "m6", 9: "M6", 10: "m7", 11: "M7" }
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=2)
    result = result + "   "+''.join([f"{intervals[i]+' ':>8}" for i in range(12)]) + '\n'
    for pitch in range(12):
        result = result + f"{pitches[pitch]:<3}"
        for interval in range(12):
            result = result + f"{intervalsizes[pitch, interval]:8.2f}"
        result = result + '\n'
    result = result + '\n'
    result = result + "avg"
    for pitch in range(12):
        result = result + f"{np.average(intervalsizes[:,pitch]):8.2f}"
    return result

def intervalSizes_str_ratios(intervalsizes, ratio_tolerance=1e-3):
    result = ''
    pitches = { 0: "C", 1: "C#", 2: "D", 3: "Eb", 4: "E", 5: "F", 6: "F#", 7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B" }
    intervals = { 0: "P1", 1: "m2", 2: "M2", 3: "m3", 4: "M3", 5: "P4", 6: "A4-d5", 7: "P5", 8: "m6", 9: "M6", 10: "m7", 11: "M7" }
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=2)
    result = result + "   "+''.join([f"{intervals[i]+' ':>11}" for i in range(12)]) + '\n'
    for pitch in range(12):
        result = result + f"{pitches[pitch]:<3}"
        for interval in range(12):
            result = result + f"{str(nsimplify(cents2ratio(intervalsizes[pitch, interval]), rational=True, tolerance=ratio_tolerance)):>11}"
        result = result + '\n'
    result = result + '\n'
    result = result + "avg"
    for pitch in range(12):
        result = result + f"{np.average(intervalsizes[:,pitch]):11.2f}"
    return result


def printIntervalSizes(intervalsizes, ratios=False, ratio_tolerance=1e-3):
    if ratios:
        print(intervalSizes_str_ratios(intervalsizes, ratio_tolerance=ratio_tolerance))
    else:
        print(intervalSizes_str(intervalsizes))

# if interval types is provided, acceptable sizes are taken from that
def plotBie(
        tuning,
        title,
        orientation='vertical', # or 'horizontal'
        intervals=[7,4,3,9],
        interval_types=None,
        saveplot=False,
        filename='plot.pdf',
        colormap=None,
    ):

    # Set smaller font size globally
    plt.rcParams['font.size'] = 8

    # Pitch class names in specified order: Ab, Eb, Bb, F, C, G, D, A, E, B, F#, C#
    pitch_classes = ['Ab', 'Eb', 'Bb', 'F', 'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#']
    pitch_indices = [8, 3, 10, 5, 0, 7, 2, 9, 4, 11, 6, 1]  # Corresponding indices in tuning

    interval_names = {
        0: 'P1',
        3: 'm3',
        4: 'M3',
        5: 'P4',
        7: 'P5',
        8: 'm6',
        9: 'M6',
        12: 'P8',
        1: 'm2',
        2: 'M2',
        6: 'A4/d5',
        10: 'm7',
        11: 'M7'
    }
    just_intonation = [
        0.0,
        111.73,
        203.91,
        315.64,
        386.31,
        498.04,
        590.22,
        701.96,
        813.69,
        884.36,
        996.09,
        1088.27
    ]
    acceptable_sizes = defaultdict(list)
    if interval_types != None:
        for k,v in interval_types.items():
            acceptable_sizes[k].extend(v['targets'])
    else:
        for ix, value in enumerate(just_intonation):
            acceptable_sizes[ix].append(value)

    # Calculate interval sizes
    sizes = {}
    for interval in intervals:
        sizes[interval] = []

        for i in pitch_indices:
            j = (i + interval) % 12
            size = tuning[j] - tuning[i] if j > i else (tuning[j] + 1200) - tuning[i]
            sizes[interval].append(size)

    # Create figure
    
    if orientation ==  'vertical':
        fig, axes = plt.subplots(1, len(intervals), figsize=(2.5*len(intervals), 4.0))
    else:
        fig, axes = plt.subplots(1, len(intervals), figsize=(3.0*len(intervals), 2.0))
        
    if colormap == None:
        colormap = ['skyblue', 'lightgreen', 'salmon'] + list(plt.cm.tab10.colors)
    
    for ix, interval in enumerate(intervals):

        # Subplot 1: Fifths
        axes[ix].bar(pitch_classes, sizes[interval], color=colormap[ix], edgecolor='black')
        for acceptable_size in acceptable_sizes[interval]:
            axes[ix].axhline(y=acceptable_size, color='red', linestyle='--', linewidth=1)
        axes[ix].set_ylabel(f'{interval_names[interval]} Size (cents)')
        axes[ix].set_title(str(interval_names[interval]) + f" ({','.join([str(a) for a in acceptable_sizes[interval]])})")
        axes[ix].set_ylim(interval*100-50,interval*100+50)
        axes[ix].grid(True, axis='y', linestyle='--', alpha=0.7)
        # axes[ix].legend()
        axes[ix].set_xticks(range(len(pitch_classes)))
        axes[ix].set_xticklabels(pitch_classes, rotation=0)

    # Dynamically calculate top margin
    # Base margin + adjustment based on number of subplots
    base_top = 0.96  # Starting point for max space
    adjustment = 0.0088 * (12 - len(intervals))  # Scale adjustment for 3 to 12 range
    top_margin = min(base_top - adjustment, 0.96)  # Cap at 0.98
    # Apply tight layout and adjust top margin
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()  # [left, bottom, right, top])
    if orientation == 'vertical':
        fig.subplots_adjust(top=top_margin)
    else:
        fig.subplots_adjust(top=0.75)

    if saveplot:
        plt.savefig(filename)
    plt.show()

    return fig, axes

def pprint_interval_data(idata, number_of_flats=3):

    # Pitch spellings for number_of_flats:
    # -6 ['B#', 'C#', 'C##', 'D#', 'D##', 'E#', 'F#', 'F##', 'G#', 'G##', 'A#']
    # -5 ['B#', 'C#', 'C##', 'D#', 'E', 'E#', 'F#', 'F##', 'G#', 'G##', 'A#']
    # -4 ['B#', 'C#', 'C##', 'D#', 'E', 'E#', 'F#', 'F##', 'G#', 'A', 'A#']
    # -3 ['B#', 'C#', 'D', 'D#', 'E', 'E#', 'F#', 'F##', 'G#', 'A', 'A#']
    # -2 ['B#', 'C#', 'D', 'D#', 'E', 'E#', 'F#', 'G', 'G#', 'A', 'A#']
    # -1 ['C', 'C#', 'D', 'D#', 'E', 'E#', 'F#', 'G', 'G#', 'A', 'A#']
    # 0 ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#']
    # 1 ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb']
    # 2 ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb']
    # 3 ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb']
    # 4 ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb']
    # 5 ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb']
    # 6 ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb']
    # 7 ['C', 'Db', 'D', 'Eb', 'Fb', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb']
    # 8 ['C', 'Db', 'D', 'Eb', 'Fb', 'F', 'Gb', 'G', 'Ab', 'Bbb', 'Bb']
    # 9 ['C', 'Db', 'Ebb', 'Eb', 'Fb', 'F', 'Gb', 'G', 'Ab', 'Bbb', 'Bb']
    # 10 ['C', 'Db', 'Ebb', 'Eb', 'Fb', 'F', 'Gb', 'Abb', 'Ab', 'Bbb', 'Bb']
    # 11 ['Dbb', 'Db', 'Ebb', 'Eb', 'Fb', 'F', 'Gb', 'Abb', 'Ab', 'Bbb', 'Bb']
    # 12 ['Dbb', 'Db', 'Ebb', 'Eb', 'Fb', 'Gbb', 'Gb', 'Abb', 'Ab', 'Bbb', 'Bb']

    allpitches = ["Fbb", "Cbb", "Gbb", "Dbb", "Abb", "Ebb", "Bbb",
                  "Fb", "Cb", "Gb", "Db", "Ab", "Eb", "Bb", 
                  "F", "C", "G", "D", "A", "E", "B",
                  "F#", "C#", "G#", "D#", "A#", "E#", "B#",
                  "F##", "C##", "G##", "D##", "A##", "E##", "B##"]
    pitches_subset = deque(allpitches[14-number_of_flats:26-number_of_flats])
    pitches_subset.rotate(-number_of_flats-1)
    pitches = {ix: pitches_subset[pitch_ix] for ix, pitch_ix in enumerate([0,7,2,9,4,11,6,1,8,3,10,5])}
    intervals = { 0: "P1", 1: "m2", 2: "M2", 3: "m3", 4: "M3", 5: "P4", 6: "A4-d5", 7: "P5", 8: "m6", 9: "M6", 10: "m7", 11: "M7" }

    print("Sorted by interval")
    p_iv = -1
    for iv in sorted(idata, key=lambda x: ((x[0][1]-x[0][0])%12, -x[2])):
        interval = (iv[0][1]-iv[0][0])%12
        if interval != p_iv:
            print(f"--- {intervals[interval]} ---")
            p_iv = interval
        print(f"{pitches[iv[0][0]]:<2} - {pitches[iv[0][1]]:<2} : {iv[2]:.4f}")

    print()
    print("Sorted by root note")
    for iv in sorted(idata, key=lambda x:x[0]):
        print(f"{pitches[iv[0][0]]:<2} - {pitches[iv[0][1]]:<2} : {iv[2]:.4f}")

    print()
    print("Sorted by weight")
    for iv in sorted(idata, key=lambda x:x[2], reverse=True):
        print(f"{pitches[iv[0][0]]:<2} - {pitches[iv[0][1]]:<2} : {iv[2]:.4f}")


def compute_temperaments_ismir2025(
    interval_data={},
    fifthboundss={},
    methods=['differential_evolution', 'dual_annealing', 'basinhopping'],
    no_runs=5, #number of runs per stage in the chain
    inherit_p = True, # use selected p from last run as p0
    new_p_every = 5, # every new_p_every run, do not chain previous p as p0. 0: never
    p0 = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]),
    result = {}, # allows to provide an dictionary to which the results will be added
    store_all_runs = False,
    overwriteResults=False,
    verbose = True,
    verbose_procedure=False,
    verbose_optimizer=False,
    verbose_objective=False,
):
    for data_name, data in interval_data.items():
        for fitfthbounds_name, fifthbounds in fifthboundss.items():
            for method in methods:
                conf_name = '_'.join([data_name, fitfthbounds_name, method])
                if verbose: print("Configuration: ", conf_name)
                if overwriteResults == False:
                    if conf_name in result.keys():
                        if verbose: print(f"{conf_name} already in result set. Skipping.")
                        continue
                result[conf_name] = {}
                #set some parameters:
                if fifthbounds != None:
                    da_i_t = 150.0 # was 70000 (for total weight 500)
                else:
                    da_i_t = 10.0 # was 5000 (for total weight 500)
                if method == 'dual_annealing':
                    if verbose: print(f"  {da_i_t=}")
                min_val = np.inf
                selected = None
                p0_selected = p0
                selected = {}
                result_runs = []
                for ix_run in range(no_runs):
                    if verbose: print(f"  Run {ix_run}")
                    if inherit_p:
                        if new_p_every > 0 and ix_run > 0:
                            if ix_run % new_p_every == 0:
                                p0_selected = p0
                                if verbose: print("    Resetting p0")
                    if verbose: print(f"    p0={np.array2string(p0_selected.round(3), separator=',', max_line_width=np.inf)}")
                    st = time.process_time()
                    result_optim = optimize_temperament(
                        data,
                        alpha_map={},
                        alpha_default=10, # was 10 for total weight 500
                        beta=2.0,
                        gamma=100.0, # 100 (was 50000 for total weight 500)
                        delta=2.0,
                        method=method,
                        bh_niter=5000,
                        bh_T=1.0, #was: 50
                        bh_stepsize=30,
                        da_maxiter=20000,
                        da_initial_temp= da_i_t,
                        de_maxiter = 10000,
                        de_popsize = 25,
                        de_tol=1e-8,
                        de_mutation=(0.7, 1.2),
                        verbose=verbose_procedure,
                        verbose_optimizer=verbose_optimizer,
                        verbose_objective=verbose_objective,
                        fifthbounds=fifthbounds,
                        p0=p0_selected,
                    )
                    et = time.process_time()
                    result_optim['runtime'] = et - st
                    if verbose: print(f"    tuning={np.array2string(result_optim['tuning'].round(3), separator=',', max_line_width=np.inf)}")
                    if verbose: print(f"    obj_val={result_optim['obj_val']}")
                    if verbose: print(f"    runtime={et-st} seconds.")
                    if result_optim['obj_val'] < min_val:
                        selected = copy.deepcopy(result_optim) # to prevent circular reference. this object is also in 'runs'
                        min_val = result_optim['obj_val']
                        if verbose: print("    Selected.")
                    if inherit_p:
                        p0_selected = result_optim['tuning']
                    if store_all_runs:
                        #convert np arrays to floats
                        result_optim['tuning'] = [float(v) for v in result_optim['tuning']]
                        result_optim['obj_val'] = float(result_optim['obj_val'])
                        result_runs.append(result_optim)
                selected['tuning'] = [float(pitch) for pitch in selected['tuning']] #np.array is not serializable
                selected['obj_val'] = sys.float_info.max if selected['obj_val'] == np.inf else float(selected['obj_val'])
                if verbose: print(f"  Storing: ({selected['tuning'], selected['obj_val']})")
                result[conf_name]['best_result'] = selected
                if store_all_runs:
                    if verbose: print(f"  Storing runs")
                    result[conf_name]['all_runs'] = result_runs
    return result

def compute_obj_val_ismir2025(
    interval_data={},
    fifthboundss={},
    p_in=None,
    p_name='',
    results={},
    verbose = True,
    verbose_procedure=False,
    verbose_optimizer=False,
    verbose_objective=False,
):
    for data_name, data in interval_data.items():
        for fitfthbounds_name, fifthbounds in fifthboundss.items():
            conf_name = '_'.join([p_name, data_name, fitfthbounds_name])
            result_optim = optimize_temperament(
                data,
                alpha_map={},
                alpha_default=10.0,
                beta=2.0,
                method='dual_annealing', #cause bounds penalty IN objective
                gamma=50000.0,
                delta=2.0,
                verbose=verbose_procedure,
                verbose_optimizer=verbose_optimizer,
                verbose_objective=verbose_objective,
                fifthbounds=fifthbounds,
                p_in=p_in,
            )
            results[conf_name] = {
                'tuning': p_in,
                'name': p_name,
                'obj_val': float(result_optim['obj_val'])
            }



