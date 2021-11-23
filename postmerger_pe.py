import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import bilby
from scipy.stats import gaussian_kde
import sys

import matplotlib as mpl
from matplotlib import rcParams, rc
# from matplotlib import rc
from astropy.table import Table

from scipy.optimize import fmin_l_bfgs_b
from scipy.interpolate import interp1d
from filelock import FileLock
import tarfile
import h5py
from scipy.signal import tukey
from copy import deepcopy
import OptionParser

COLWIDTH = 246.0

SCALE = 2.0

fontsize = 10 * SCALE


def get_figsize(wf=1.0, hf=(5.**0.5-1.0)/2.0):
    """Parameters:
      - wf [float]:  width fraction in columnwidth units
      - hf [float]:  height fraction in columnwidth units.
                     Set by default to golden ratio.
      - COLWIDTH [float]: width of the column in latex. Get this from LaTeX 
                             using \showthe\columnwidth
    Returns:  [fig_width,fig_height]: that should be given to matplotlib
    """
    fig_width_pt = COLWIDTH*wf
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*hf      # height in inches
    return fig_width, fig_height


def main():

    fontparams = {'mathtext.fontset': 'stix',
                  'font.family': 'serif',
                  'font.serif': "Times New Roman",
                  'mathtext.rm': "Times New Roman",
                  'mathtext.it': "Times New Roman:italic",
                  'mathtext.sf': 'Times New Roman',
                  'mathtext.tt': 'Times New Roman'}
    rcParams.update(fontparams)

    rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    options = OptionParser.get_options()
    print(options)

    if 'None' in options.labelbase:
        label_base = sys.argv[0][:-3].split('/')[-1]
    else:
        label_base = options.labelbase
    print("Hello")

    print(f'Filename = {label_base}')
    print(f'Waveform order = {options.waveform_order}')
    print(f'SNR = {options.snr:0.3e}')
    print(f'SEED = {options.seed:d}')
    print(f'Number of points = {options.npoints}')
    print(f'Delta function t_0: {options.t0delta}')
    print(f'Zero noise: {options.zero_noise}')

    outdir = options.plotdir

    loudness = 2.512e-20 * options.snr / 50

    options.t0search = options.deltat0searchx10 / 10.0 + options.t0ms

    if not options.t0delta:
        label = (f'{label_base}_order_{options.waveform_order}_SNR_{options.snr:0.2f}_aLIGO'
                    f'_0noise_{options.zero_noise}_NRfile_{options.nrwaveform}_log_nlive{options.npoints}'
                    f'_walks{options.nwalkers}_t0search_matchedinput{options.matchedinput}_seed{options.seed}'
                    f'_job{options.jobid}')
    else:
        label = (f'{label_base}_order_{options.waveform_order}_SNR_{options.snr:0.2e}_aLIGO'
                    f'_0noise_{options.zero_noise}_NRfile_{options.nrwaveform}_log_nlive{options.npoints}'
                    f'_walks{options.nwalkers}_injectedt0{options.t0ms}_adjustedt0{options.t0search:0.1f}'
                    f'_matchedinput{options.matchedinput}_seed{options.seed}_job{options.jobid}temp')

    print(f'Plot directory:\n{options.plotdir}')
    print(f'Output directory:\n{outdir}')
    print(f'Label:\n{label}')

    # create plots directory if it does not exist
    os.makedirs(options.plotdir, exist_ok=True)
    

    def inner_product(aa, bb, freq, psd, model_variance):
        """
        Calculate the inner product defined in the matched filter statistic

        arguments:
        aai, bb: single-sided Fourier transform, created, e.g., by the nfft function above
        freq: an array of frequencies associated with aa, bb, also returned by nfft
        psd: noise power spectral density at frequencies bins in freq
        model_variance : is the variance of the model, same bins as freq

        Returns:
        The matched filter inner product for aa and bb
        """

        # calculate the inner product
        integrand = np.conj(aa) * bb / (psd + model_variance)

        df = freq[1] - freq[0]
        integral = np.sum(integrand) * df

        product = 4. * np.real(integral)

        return product

    def fitting_factor(strain1, strain2, frequency, psd, model_variance):
        """
        fitting_factor = <strain1,strain2>/sqrt(<strain1,strain1><strain2,strain2>)
        the input is the STRAIN = strain (not characteristic strain)
        Make sure that the psd and model_variance are scaled to the same amplitude
        """
        return (inner_product(
            strain1,
            strain2,
            frequency, psd,
            np.array(model_variance).astype(float)) / np.sqrt(
            inner_product(strain1,
                            strain1,
                            frequency, psd,
                          np.array(model_variance).astype(float)) *
            inner_product(strain2,
                            strain2,
                            frequency, psd,
                            np.array(model_variance).astype(float))))

    class GRutils:

        def __init__(self):
            self.cc = 299792458.0  # speed of light in m/s
            self.GG = 6.67384e-11  # Newton in m^3 / (kg s^2)
            self.Msun = 1.98855 * 10 ** 30  # solar mass in  kg
            self.kg = 1. / self.Msun
            self.metre = self.cc ** 2 / (self.GG * self.Msun)
            self.secs = self.cc * self.metre
            self.Mpc = 3.08568e+22  # Mpc in metres

        def m_sol_to_geo(self, mm):
            # convert from solar masses to geometric units
            return mm / self.kg * self.GG / self.cc ** 2

        def dist_Mpc_to_geo(self, dist):
            # convert distance from Mpc to geometric units (i.e., metres)
            return dist * self.Mpc

        def time_geo_to_s(self, time):
            # convert time from seconds to geometric units
            return time / self.cc

        @staticmethod
        def nfft(ht, Fs):
            """
            performs an FFT while keeping track of the frequency bins
            assumes input time series is real (positive frequencies only)

            ht = time series
            Fs = sampling frequency

            returns
            hf = single-sided FFT of ft normalised to units of strain / sqrt(Hz)
            f = frequencies associated with hf
            """
            # add one zero padding if time series does not have even number of sampling times
            if np.mod(len(ht), 2) == 1:
                ht = np.append(ht, 0)
            base_length = len(ht)
            # frequency range
            ff = Fs / 2 * np.linspace(0, 1, int(round(base_length / 2 + 1)))
            # calculate FFT
            # rfft computes the fft for real inputs
            hf = np.fft.rfft(ht)

            # normalise to units of strain / Hz
            hf = hf / Fs

            return hf, ff

    def generate_parameter_names(waveform_order):
        parameter_names = [
            f'{x}_{y}'
                for x in ['f', 'T', 'phi', 'alpha']
                for y in range(waveform_order)] + ['t_0', 'logB', 'w_0', 'w_1']
        return parameter_names

    def create_index_dictionary_from_sorted_amplitudes(waveform_order, amplitudes):

        initial_indices = np.arange(0, waveform_order)
        final_indices = np.flip(np.argsort(amplitudes[:waveform_order]), 0)

        return {initial_index: final_index for (initial_index, final_index) in zip(initial_indices, final_indices)}

    def create_zero_dictionary(keys):

        return {key: 0.0 for key in keys}

    def calculate_total_log_amplitude(log_Amplitude_array, waveform_order):

        return np.sum(log_Amplitude_array[:waveform_order])

    def create_injection_values(parameter_names, waveform_order, injectionvalues):

        df_injection_values = pd.read_csv(
            injectionvalues, header=None, index_col=0)
        frequencies = df_injection_values.loc['frequencies', :].values
        log_amplitudes = df_injection_values.loc['log_amplitudes', :].values
        decay_time_constants = df_injection_values.loc['decay_time_constants', :].values
        phases = df_injection_values.loc['phases', :].values
        alphas = df_injection_values.loc['alphas', :].values

        # want to select values in order from the above lists, but also then want to sort the
        # parameter numbers so that amplitudes are decreasing, hence A_{i} > A_{i+1} > A_{i+2}
        # The following dictionary should do the job

        index_dict = create_index_dictionary_from_sorted_amplitudes(
            waveform_order, log_amplitudes)

        injection_values_dict = create_zero_dictionary(parameter_names)
        total_logA = calculate_total_log_amplitude(
            log_amplitudes, waveform_order)
        for waveform_parameter_name in parameter_names:
            if waveform_parameter_name == 'logB':
                injection_values_dict[waveform_parameter_name] = np.log10(
                    loudness)
            else:
                waveform_parameter_type, waveform_parameter_number = waveform_parameter_name.split(
                    '_')
                waveform_parameter_number = index_dict[int(
                    waveform_parameter_number)]
                if waveform_parameter_type == 'w':
                    injection_values_dict[waveform_parameter_name] = log_amplitudes[
                        waveform_parameter_number] - total_logA
                elif waveform_parameter_type == 'f':
                    injection_values_dict[waveform_parameter_name] = frequencies[waveform_parameter_number]
                elif waveform_parameter_type == 'T':
                    injection_values_dict[waveform_parameter_name] = decay_time_constants[waveform_parameter_number]
                elif waveform_parameter_type == 'phi':
                    injection_values_dict[waveform_parameter_name] = phases[waveform_parameter_number]
                elif waveform_parameter_type == 'alpha':
                    injection_values_dict[waveform_parameter_name] = alphas[waveform_parameter_number]
        return injection_values_dict

    def hfmax(parameters, mode_number):
        """
        hfmax(parameters,mode_number)
        parameters: must be passed from the prior conversion function
        mode_number: which waveform is to be calculated, for a three waveform system mode_number is 0 or 1 or 2
        This function calculates the approximate max FFT peak for a single mode given the dependent variables
        A, T and alpha"""

        # print(locals())
        # 1/0
        plus = np.zeros(len(GLOBAL_TIME))
        #     t_0 = t_0
        #     print(t_0)
        if options.t0delta:
            dt = GLOBAL_TIME - options.t0search / 1000.0
        else:
            dt = GLOBAL_TIME - parameters['t_0'] / 1000.0
        tpos = dt[dt >= 0]
        dplus = np.zeros(len(tpos))
        Two_pi_dt = 2 * np.pi * tpos
        dtpos = tpos[-1] - tpos[0]
        window = tukey(len(tpos), 2 * options.tukeyrolloffms /
                    dtpos / 1000.0)  # 0.2ms tukey

        if mode_number == 2:
            A = (10 ** parameters['logB']) * (1 - parameters['w01'])
        else:
            A = (10 ** parameters['logB']) * parameters[f'w_{mode_number}']
        dplus, _ = calculate_mode_waveform(parameters[f'f_{mode_number}'],
                                            parameters[f'T_{mode_number}'] / 1000.0,
                                            parameters[f'phi_{mode_number}'],
                                            parameters[f'alpha_{mode_number}'],
                                            A, tpos, Two_pi_dt, dplus, None)
        plus[dt >= 0] = dplus * window

        gr = GRutils()
        hf, ff = gr.nfft(plus, 16384)
        #     print(sum(plus))
        return np.log10(np.max(np.abs(hf) * (ff ** options.hfmaxfreqscale)))

    def convert_parameters(parameters):
        """
        Function to convert between sampled parameters and constraint parameter.

        Parameters
        ----------
        parameters: dict
            Dictionary containing sampled parameter values, 'f_i'.. 'f_j'.
            Dictionary containing sampled parameter values, 'logA_i'.. 'logA_j'.

        Returns
        -------
        dict: Dictionary with constraint parameter
        'zf_i' added with zf_i = |f_i - f_0|
        'za_i' added with za_i = logA_0 - logA_i
        for all i > 0.
        """

        # amplitudes are now sorted in descending order
        parameters['w01'] = parameters['w_0'] + parameters['w_1']
        parameters['hf_0'] = hfmax(parameters, 0)
        parameters['hf_1'] = hfmax(parameters, 1)
        parameters['hf_2'] = hfmax(parameters, 2)
        parameters['hf01'] = parameters['hf_0'] - parameters['hf_1']
        parameters['hf12'] = parameters['hf_1'] - parameters['hf_2']

        return parameters

    def create_priors(injection_parameters):
        prior = bilby.core.prior.PriorDict(
            conversion_function=convert_parameters)
        prior.update(injection_parameters.copy())
        for waveform_parameter_name in injection_parameters.keys():
            if waveform_parameter_name == 'logB':
                prior[waveform_parameter_name] = bilby.core.prior.Uniform(
                    -24, -17, r'$logB$')
            else:
                waveform_parameter_type, waveform_parameter_number = waveform_parameter_name.split(
                    '_')
                if waveform_parameter_type == 'f':
                    waveform_parameter_number = int(waveform_parameter_number)
                    prior[waveform_parameter_name] = bilby.core.prior.Uniform(1000, 5000,
                                                                            fr'$f_{{{waveform_parameter_number:d}}}$')
                elif waveform_parameter_type == 'T':
                    waveform_parameter_number = int(waveform_parameter_number)
                    prior[waveform_parameter_name] = bilby.core.prior.LogUniform(0.1, 2000,
                                                                                fr'$T_{{{waveform_parameter_number:d}}}$')
                elif waveform_parameter_type == 'phi':
                    waveform_parameter_number = int(waveform_parameter_number)
                    prior[waveform_parameter_name] = bilby.core.prior.Uniform(-np.pi, np.pi,
                                                                            fr'$\phi_{{{waveform_parameter_number:d}}}$',
                                                                            boundary='periodic')
                elif waveform_parameter_type == 'alpha':
                    waveform_parameter_number = int(waveform_parameter_number)
                    prior[waveform_parameter_name] = bilby.core.prior.Uniform(-6.4, 6.4,
                                                                            fr'$\alpha_{{{waveform_parameter_number:d}}}$')  # alpha_0, alpha_1
                elif waveform_parameter_type == 'w':
                    waveform_parameter_number = int(waveform_parameter_number)
                    prior[waveform_parameter_name] = bilby.core.prior.Uniform(0, 1,
                                                                            fr'$w_{{{waveform_parameter_number:d}}}$')  # w_0, w_1
                elif waveform_parameter_type == 't':  # this is for t_0
                    if options.t0delta:
                        prior['t_0'] = bilby.core.prior.DeltaFunction(
                            options.t0search, r'$t_0$')
                    else:
                        prior['t_0'] = bilby.core.prior.Uniform(options.t0ms - options.t0searchrangems,
                                                                options.t0ms + options.t0searchrangems, r'$t_0$')
        prior['hf01'] = bilby.core.prior.Constraint(
            0.0, 10, r'$\Delta hf_{01}$')
        prior['hf12'] = bilby.core.prior.Constraint(
            0.0, 10, r'$\Delta hf_{12}$')
        prior['w01'] = bilby.core.prior.Constraint(0.0, 1, r'$\Delta w_{01}$')
        return prior

    def numerical_relativity_postmerger_waveform(newtime, **waveform_kwargs):

        tar_location = waveform_kwargs['nr_tar_path']
        filename = waveform_kwargs['filename']
        loudness = waveform_kwargs['loudness']
        t_inspiral_ms = waveform_kwargs['t_inspiral_ms']
        tukey_rolloff = waveform_kwargs['tukey_rolloff']

        t_0 = waveform_kwargs['t_0'] / 1000.0

        if not tar_location.endswith('/'):
            tar_location += '/'
        lock = FileLock(options.lock_filename)
        with lock:
            with tarfile.open(tar_location + filename, 'r') as tar:
                metadatalocn = tar.getmember(tar.getnames()[1])
                # this is the location of the hdf5 portion of the tar
                h5locn = tar.getmember(tar.getnames()[2])
                h5locn.name = tar_location + h5locn.name
                tar.extract(h5locn)
                with h5py.File(h5locn.name, 'r') as f:
                    rh22 = f['rh_22']
                    keys = list(rh22.keys())
                    h22furthest = [key for key in keys if (
                        'Rh_l2_m2' in key) and not ('Inf' in key)][-1]
                    rh22data = rh22[h22furthest]
                    hr_pl_msun = rh22data[:, 1] * 2 * 2 ** 0.5 # This scaling is an approximation
                    hr_cr_msun = rh22data[:, 2] * 2 * 2 ** 0.5 # This scaling is an approximation
                    time_msun = rh22data[:, 8]
        full_wave = (hr_pl_msun ** 2 + hr_cr_msun ** 2) ** 0.5
        postmerger_start_index = np.argmax(full_wave)
        gr = GRutils()
        hscale = loudness * 1.2 / np.max([
            np.max(np.abs(hr_pl_msun[postmerger_start_index:])),
            np.max(np.abs(hr_cr_msun[postmerger_start_index:]))])  # max value should be ~max(|h+|,|hx|)
        tscale = gr.time_geo_to_s(gr.m_sol_to_geo(1))

        hre = hr_pl_msun * hscale
        him = hr_cr_msun * hscale
        # th = 0 @ merger
        th = (time_msun - time_msun[postmerger_start_index]) * tscale
        tstartindex = np.argmax(th > - t_inspiral_ms / 1000.0)
        hrenew = hre[tstartindex:]
        himnew = him[tstartindex:]
        thnew = th[tstartindex:] + t_0  # postmerger started @ t=0 now @ t_0

        hplus_interp_func = interp1d(thnew,
                                    hrenew,
                                    bounds_error=False, fill_value=0)

        hcross_interp_func = interp1d(thnew,
                                    himnew,
                                    bounds_error=False, fill_value=0)

        time = newtime - newtime[0]
        hplus = hplus_interp_func(time)
        hcross = hcross_interp_func(time)

        #     redefine tstartindex based on interpolated data
        tstartindex = np.argmax(time >= t_0 - t_inspiral_ms / 1000.0)
        tout = time[tstartindex:]
        pos_length = len(tout)
        dtpos = tout[-1] - tout[0]
        hplus = np.hstack(
            [np.zeros(len(hplus[:tstartindex])), hplus[tstartindex:] * tukey(pos_length, 2 * tukey_rolloff / dtpos)])
        hcross = np.hstack(
            [np.zeros(len(hcross[:tstartindex])), hcross[tstartindex:] * tukey(pos_length, 2 * tukey_rolloff / dtpos)])

        return {'plus': hplus, 'cross': hcross}

    def calculate_mode_waveform(f, T, phi, alpha, A, tpos, Two_pi_dt, dplus, dcross):

        Amp = np.exp(-tpos / T) * A
        angle = Two_pi_dt * f * (1 + alpha * tpos) + phi
        if dplus is not None:
            dplus = Amp * np.sin(angle)
        if dcross is not None:
            dcross = Amp * np.cos(angle)
        return dplus, dcross

    # define the time-domain model
    def time_domain_damped_sinusoid(time, **kwargs):
        """
        Three damped sinusoidal waveforms, h+:cos, hx:sin.
        """

        #     print('****',kwargs)

        waveform_order = search_waveform.waveform_arguments['waveform_order']
        tukey_rolloff = search_waveform.waveform_arguments['tukey_rolloff']

        plus = np.zeros(len(time))
        cross = plus.copy()
        t_0 = kwargs['t_0'] / 1000.0
        #     print(t_0)
        dt = time - t_0
        tpos = dt[dt >= 0]
        dplus = np.zeros(len(tpos))
        dcross = dplus.copy()
        Two_pi_dt = 2 * np.pi * tpos
        dtpos = tpos[-1] - tpos[0]
        window = tukey(len(tpos), 2 * tukey_rolloff / dtpos)  # 0.2ms tukey
        for waveform_number in range(waveform_order):
            if waveform_number == 2:
                A = (1 - kwargs['w_0'] - kwargs['w_1']) * 10 ** kwargs['logB']
            else:
                A = kwargs[f'w_{waveform_number}'] * 10 ** kwargs['logB']
            f = kwargs[f'f_{waveform_number}']
            T = kwargs[f'T_{waveform_number}'] / 1000.0
            phi = kwargs[f'phi_{waveform_number}']
            alpha = kwargs[f'alpha_{waveform_number}']
            Amp = np.exp(-tpos / T) * A
            # note alpha is multiplied by (t-t_0)**2
            angle = Two_pi_dt * f * (1 + alpha * tpos) + phi
            ddplus, ddcross = calculate_mode_waveform(kwargs[f'f_{waveform_number}'],
                                                        kwargs[f'T_{waveform_number}'] / 1000.0,
                                                        kwargs[f'phi_{waveform_number}'],
                                                        kwargs[f'alpha_{waveform_number}'],
                                                        A, tpos, Two_pi_dt, dplus, dcross)
            dplus += ddplus
            dcross += ddcross
        plus[dt >= 0] = dplus * window
        cross[dt >= 0] = dcross * window
        #     print(sum(plus))
        # plt.figure()
        # plt.plot(time,plus)
        # plt.xlim(0.032, 0.042)
        # plt.show()
        # 1/0
        return {'plus': plus, 'cross': cross}

    t_0 = options.t0ms
    # approx_duration = 0.025 + t_0
    sampling_frequency = 8192 * 2
    duration = options.durationms / 1000.0

    GLOBAL_TIME = np.arange(0, duration, 1 / sampling_frequency)

    waveform_parameter_names = generate_parameter_names(
        options.waveform_order)  # used for priors only

    function_parameters = create_injection_values(waveform_parameter_names,
                                                    options.waveform_order,
                                                    options.injectionvaluesfile)  # used for priors only

    priors = create_priors(function_parameters)

    injection_parameters = (dict(ra=0.0, dec=0.0, psi=0.0, geocent_time=0.0))

    function_parameters['t_0'] = t_0
    injection_parameters['t_0'] = t_0

    priors.update(dict(ra=0.0, dec=0.0, psi=0.0, geocent_time=0.0))

    def calculate_network_snr(snrs):

        return sum(snrs ** 2) ** 0.5

    def SetSourceWaveform(waveform_arguments, seed, zero_noise, fmin, fmax, duration, sampling_frequency,
                            time_domain_source_model,
                            parameters,
                            start_time):

        waveform = bilby.gw.WaveformGenerator(
            duration=duration, sampling_frequency=sampling_frequency,
            time_domain_source_model=time_domain_source_model,
            waveform_arguments=waveform_arguments, parameters=parameters,
            start_time=start_time)

        np.random.seed(seed)  # run this before
        ifos = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
        for interferometer in ifos:
            interferometer.minimum_frequency = fmin
            interferometer.maximum_frequency = fmax

        np.random.seed(seed)  # run this before
        ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=sampling_frequency, duration=duration,
            start_time=start_time)

        for interferometer in ifos:
            if zero_noise:
                interferometer.set_strain_data_from_zero_noise(
                    sampling_frequency=sampling_frequency, duration=duration)
        np.random.seed(seed)  # run this before
        ifos.inject_signal(waveform_generator=waveform,
                            parameters=injection_parameters)
        # plt.figure()
        # plt.plot(waveform.time_array,waveform.time_domain_strain()['plus'])
        # plt.xlim(0.032,0.042)
        # plt.show()

        snrs = np.array([interferometer.meta_data['optimal_SNR']
                        for interferometer in ifos])
        network_snr = calculate_network_snr(snrs)

        return network_snr, snrs, ifos, waveform

    def print_snr_data(message, network_snr, snr_array, injections):

        print(message)
        print(f'SNRs = {snr_array}')
        print(f'Network SNR = {network_snr:0.3f}')
        print(f'Injection parameters = {injections}')

    source_waveform_parameters = dict(duration=duration, sampling_frequency=sampling_frequency,
                                        time_domain_source_model=numerical_relativity_postmerger_waveform,
                                        parameters=injection_parameters,
                                        start_time=injection_parameters['geocent_time'])

    # Calculate post-merger SNR only

    source_waveform_arguments = dict(nr_tar_path=options.nrpath, filename=options.nrwaveform, loudness=loudness,
                                    t_0=t_0,
                                    t_inspiral_ms=0.0, tukey_rolloff=options.tukeyrolloffms / 1000.0)

    network_snr, snrs, ifos, source_waveform = SetSourceWaveform(source_waveform_arguments, options.seed,
                                                                    options.zero_noise, options.ifominfrequency,
                                                                    options.ifomaxfrequency,
                                                                    **source_waveform_parameters)

    print_snr_data('Initial SNR', network_snr, snrs, injection_parameters)

    # Need to adjust the SNR so that network_snr = SNR

    loudness = loudness * options.snr / network_snr

    # Scale to correct post-merger SNR and print resulting SNR

    source_waveform_arguments = dict(nr_tar_path=options.nrpath, filename=options.nrwaveform, loudness=loudness,
                                        t_0=t_0,
                                        t_inspiral_ms=0.0, tukey_rolloff=options.tukeyrolloffms / 1000.0)

    network_snr, snrs, ifos, source_waveform = SetSourceWaveform(source_waveform_arguments, options.seed,
                                                                    options.zero_noise, options.ifominfrequency,
                                                                    options.ifomaxfrequency,
                                                                    **source_waveform_parameters)

    print_snr_data(
        f'Scaled to post-merger SNR of {options.snr:0.1f}', network_snr, snrs, injection_parameters)

    # Remove TSEARCH from start of waveform

    if options.matchedinput:
        source_waveform_arguments = dict(nr_tar_path=options.nrpath, filename=options.nrwaveform, loudness=loudness,
                                            t_0=t_0,
                                            t_inspiral_ms=t_0 - options.t0search,
                                            tukey_rolloff=options.tukeyrolloffms / 1000.0)

        network_snr, snrs, ifos, source_waveform = SetSourceWaveform(source_waveform_arguments, options.seed,
                                                                        options.zero_noise, options.ifominfrequency,
                                                                        options.ifomaxfrequency,
                                                                        **source_waveform_parameters)

        print_snr_data(f'Scaled to post-merger SNR of {options.snr:0.1f}, started at {options.t0search:0.1f}ms',
                        network_snr, snrs, injection_parameters)

    # make search waveform generator

    search_waveform = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        time_domain_source_model=time_domain_damped_sinusoid,
        start_time=injection_parameters['geocent_time'])

    search_waveform.source_parameter_keys = set(function_parameters.keys())
    search_waveform.parameters = function_parameters.copy()
    search_waveform.waveform_arguments = dict(waveform_order=options.waveform_order,
                                                tukey_rolloff=options.tukeyrolloffms / 1000.0)

    # prior['t0'] = bilby.core.prior.Uniform(0, 0.025, r'$t_0$')

    # define likelihood
    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        ifos, search_waveform)
    # launch sampler
    if options.maxiter:
        result = bilby.core.sampler.run_sampler(  # quick and nasty
            likelihood, priors, sampler='dynesty', npoints=options.npoints, maxiter=options.maxiter, dlogz=400.0,
            injection_parameters=injection_parameters, outdir=outdir,  # nthreads=8, maxmcmc=3000,
            label=label, check_point_delta_t=600, resume=options.resume, verbose=1)
    else:
        result = bilby.core.sampler.run_sampler(
            likelihood, priors, sampler='dynesty', npoints=options.npoints, walks=options.nwalkers,
            injection_parameters=injection_parameters, outdir=outdir,  # nthreads=8, maxmcmc=3000,
            label=label, resume=options.resume, verbose=1, n_check_point=70000)

    label = f'{label}_LogBF{result.log_bayes_factor:0.2f}'
    corner_parameters_wanted = set(function_parameters)
    if options.t0delta:
        if 't_0' in corner_parameters_wanted:
            # don't plot delta function in posteriors
            corner_parameters_wanted.remove('t_0')

    corner_parameters = sorted(list(corner_parameters_wanted))
    # corner_truths = [function_parameters[key] for key in corner_parameters]

    for order in range(options.waveform_order):
        result.posterior[f'msT_{order}'] = result.posterior[f'T_{order}']

    if not options.t0delta:
        result.posterior['mst_0'] = result.posterior['t_0']
    new_corner_parameters = [param if param not in ['T_0', 'T_1', 'T_2', 't_0'] else 'ms' + param for param in
                                corner_parameters]
    print('Corner parameters\n', new_corner_parameters)

    fig = result.plot_corner(new_corner_parameters)

    # fig.suptitle(label, fontsize=20)
    fig.savefig(f'{options.plotdir}{label}_corner.pdf')


if __name__ == '__main__':
    main()
