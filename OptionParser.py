import configargparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')

def get_options():


    p = configargparse.ArgParser(default_config_files=['nrexinspiral_2msstart_nomatched_input.conf'])

    p.add_argument(
        "--config_filename", type=str, is_config_file=True
    )

    p.add_argument(
        "--nrpath", type=str,
    )

    p.add_argument(
        "--lock_filename", type=str,
    )

    p.add_argument(
        "--nrwaveform", type=str,
    )

    p.add_argument(
        "--waveform_order", type=int,
    )

    p.add_argument(
        "--snr", type=float,
    )

    p.add_argument(
        "--seed", type=int,
    )

    p.add_argument(
        "--npoints", type=int,
    )

    p.add_argument(
        "--nwalkers", type=int,
    )

    p.add_argument(
        "--alpha_on", type=str2bool,
    )

    p.add_argument(
        "--t0delta", type=str2bool,
    )

    p.add_argument(
        "--t0ms", type=float,
    )

    p.add_argument(
        "--deltat0searchx10", type=int,
    )

    p.add_argument(
        "--zero_noise", type=str2bool,
    )

    p.add_argument(
        "--maxiter", type=str2bool,
    )

    p.add_argument(
        "--tinspiralms", type=float,
    )

    p.add_argument(
        "--tukeyrolloffms", type=float,
    )

    p.add_argument(
        "--durationms", type=float,
    )

    p.add_argument(
        "--t0searchrangems", type=float,
    )

    p.add_argument(
        "--ifominfrequency", type=float,
    )

    p.add_argument(
        "--ifomaxfrequency", type=float,
    )

    p.add_argument(
        "--plotdir", type=str,
    )

    p.add_argument(
        "--jobid", type=str,
    )

    p.add_argument(
        "--resume", type=str2bool,
    )

    p.add_argument(
        "--matchedinput", type=str2bool,
    )

    p.add_argument(
        "--injectionvaluesfile", type=str,
    )

    p.add_argument(
        "--labelbase", type=str,
    )

    p.add_argument(
        "--hfmaxfreqscale", type=float,
    )
    return p.parse_args()