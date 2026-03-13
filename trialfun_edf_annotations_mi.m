function [trl, event] = trialfun_edf_annotations_mi(cfg)
% Return TRL (Nx4): [begSample endSample offset code]
% code: 1 = left, 2 = right

    hdr   = ft_read_header(cfg.dataset);
    event = ft_read_event(cfg.dataset, 'detectflank', 'both');

    fs    = hdr.Fs;
    preS  = round(cfg.trialdef.pre  * fs);
    postS = round(cfg.trialdef.post * fs);

    % Accept both plain and prefixed labels
    leftSet  = string(cfg.codes.left);   % e.g., {'Trigger#21','21'}
    rightSet = string(cfg.codes.right);  % e.g., {'Trigger#22','22'}

    trl = zeros(0,4);

    for i = 1:numel(event)
        s = string(event(i).value);
        % also match trailing digits: 'Trigger#21' -> '21'
        tok = regexp(char(s), '(\d+)$', 'tokens');
        if ~isempty(tok), s_num = string(tok{1}{1}); else, s_num = s; end

        code = 0;
        if any(s == leftSet)  || any(s_num == leftSet),  code = 1; end
        if any(s == rightSet) || any(s_num == rightSet), code = 2; end
        if code == 0, continue; end

        samp = event(i).sample;
        if isempty(samp) || isnan(samp), continue; end

        begS = max(1, samp - preS);
        endS = min(hdr.nSamples * hdr.nTrials, samp + postS - 1);
        if endS <= begS, continue; end

        ofsS = -preS;
        trl(end+1,:) = [begS endS ofsS code]; %#ok<AGROW>
    end

    if isempty(trl)
        error('No trials matched codes %s / %s. Check event labels.',
            strjoin(cellstr(leftSet),','), strjoin(cellstr(rightSet),','));
    end
end
