# Postmerger PE code

To make this work you need to use Bilby version 0.5.6

`pip install bilby==0.5.6`

You also need to download the numerical-relativity simulation files that you want from:

`https://core-gitlfs.tpi.uni-jena.de/core_database/`

For instance, the code listed here used the SLy THC:0036:R03 simulation listed here:

`https://core-gitlfs.tpi.uni-jena.de/core_database/THC_0036/-/tree/master/R03`

and place the `THC:0036:R03.tar` file in the `./NRtars` directory.

Also a dependency, tables, does not work with python `3.9` onwards. I have tested this with python `3.7.2`.

Example script to run the python file is `example_run.sh`.

deltat0searchx10 sets the time delay, in units of 0.1ms, for the start of the waveform model after the merger time. For example if deltat0searchx10 = 10, then the waveform model won't start until 1ms after the merger time. Usually this is set to 0.

hfmaxfreqscale should usually be set to `0.5`. When the frequency peaks are sorted by amplitude, the amplitude is usually measured as `|H(f)|sqrt(f)`. Changing this value from `0.5` will result in `|H(f)| * (f ** hfmaxfreqscale)`.

snr sets the postmerger snr.

Other default values are stored in the configuration file: `nrexinspiral_2msstart.conf` and any value can be overriden in the command line as shown in `example_run.sh`.

See `https://arxiv.org/abs/2006.04396` for more details.