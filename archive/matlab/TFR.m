%% ===============================================================
%  MI TFR from EDF(+): annotations 21=left, 22=right
%% ===============================================================
clear; clc;
addpath('C:\Users\user\AppData\Roaming\MathWorks\MATLAB Add-Ons\Collections\FieldTrip')
ft_defaults

% --- EDIT: file path
dataset = fullfile('data', 'NIC2', '20250820161249_img_multi', '20250820161249_img_multi_Experiment.edf');  % <= change to your EDF(+)

% --- EDIT: analysis parameters
channels_of_interest = {'C3','C4'};   % or 'all'
trial_pre   = 1.0;      % s before event
trial_post  = 3.0;      % s after event
baseline_win = [-0.8 0];
toi         = -1.0:0.02:3.0;
foi         = 2:1:40;

tfr_method  = 'wavelet';  % 'wavelet' (Morlet) or 'mtmconvol'
morlet_width = 7;

% multitaper (only used if tfr_method='mtmconvol')
taper        = 'hanning';
t_ftimwin    = 5 ./ foi; % ~5 cycles per frequency

% Preproc toggles
do_reref   = true;  refchannel = 'all';
do_bp      = true;  bpfreq = [1 40];
do_notch   = true;  notch = 50;
target_fs  = 250;

%% 1) Define trials from EDF annotations (21=left, 22=right)
cfg = [];
cfg.dataset        = dataset;           % your EDF path
cfg.trialfun       = 'trialfun_edf_annotations_mi';
cfg.trialdef.pre   = 1.0;               % seconds before cue
cfg.trialdef.post  = 3.0;               % seconds after cue
cfg.codes.left     = {'Trigger#21','21'};
cfg.codes.right    = {'Trigger#22','22'};

cfg = ft_definetrial(cfg);              % <-- now returns cfg.trl
%% 2) Preprocessing
cfg_prep = [];
cfg_prep.dataset  = dataset;
cfg_prep.trl      = cfg.trl;
cfg_prep.channel  = channels_of_interest;

if do_bp,   cfg_prep.bpfilter='yes'; cfg_prep.bpfreq=bpfreq; else, cfg_prep.bpfilter='no'; end
if do_notch && ~isempty(notch)
    cfg_prep.bsfilter='yes'; cfg_prep.bsfreq=[notch-1 notch+1];
end
if do_reref
    cfg_prep.reref='yes'; cfg_prep.refmethod='avg'; cfg_prep.refchannel=refchannel;
end
cfg_prep.demean='yes';
data = ft_preprocessing(cfg_prep);

if ~isempty(target_fs)
    cfg_rs=[]; cfg_rs.resamplefs=target_fs; cfg_rs.detrend='no';
    data = ft_resampledata(cfg_rs, data);
end

%% 3) Time-Frequency
cfg_tfr = [];
cfg_tfr.output='pow'; cfg_tfr.method=tfr_method;
cfg_tfr.foi=foi; cfg_tfr.toi=toi; cfg_tfr.keeptrials='yes';
switch lower(tfr_method)
    case 'wavelet',    cfg_tfr.width=morlet_width;
    case 'mtmconvol',  cfg_tfr.taper=taper; cfg_tfr.t_ftimwin=t_ftimwin;
end
tfr = ft_freqanalysis(cfg_tfr, data);

%% 4) Baseline to dB
cfg_bl=[]; cfg_bl.baseline=baseline_win; cfg_bl.baselinetype='db';
tfr_bl = ft_freqbaseline(cfg_bl, tfr);

%% 5) Split & average
isLeft  = (data.trialinfo==1);
isRight = (data.trialinfo==2);

tfr_left  = ft_selectdata(struct('trials', find(isLeft)),  tfr_bl);
tfr_right = ft_selectdata(struct('trials', find(isRight)), tfr_bl);

tfrL_avg = ft_freqdescriptives([], tfr_left);
tfrR_avg = ft_freqdescriptives([], tfr_right);

%% 6) Plot TFRs for each channel
for ch = 1:numel(tfrL_avg.label)
    chan = tfrL_avg.label{ch};
    figure('Name',['Left MI - ' chan],'Color','w');
    ft_singleplotTFR(struct('channel',chan,'zlim','maxabs'), tfrL_avg);
    title(['Left MI (dB): ' chan]);

    figure('Name',['Right MI - ' chan],'Color','w');
    ft_singleplotTFR(struct('channel',chan,'zlim','maxabs'), tfrR_avg);
    title(['Right MI (dB): ' chan]);
end

%% 7) Quick ERD/ERS summary (μ:8–13, β:13–30, 0.5–2.0 s)
mu=[8 13]; be=[13 30]; tw=[0.5 2.0];
fprintf('\n--- ERD (dB) summary ---\n');
for ch=1:numel(tfrL_avg.label)
    lab=tfrL_avg.label{ch};
    sel = @(f,t) squeeze(ft_selectdata(struct('channel',lab,'frequency',f,'latency',t,'avgoverfreq','yes','avgovertime','yes'), tfr_left).powspctrm);
    muL = mean(sel(mu,tw),'all'); beL = mean(sel(be,tw),'all');
    sel = @(f,t) squeeze(ft_selectdata(struct('channel',lab,'frequency',f,'latency',t,'avgoverfreq','yes','avgovertime','yes'), tfr_right).powspctrm);
    muR = mean(sel(mu,tw),'all'); beR = mean(sel(be,tw),'all');
    fprintf('%s  μ: L %.2f dB | R %.2f dB   β: L %.2f dB | R %.2f dB\n', lab, muL, muR, beL, beR);
end

