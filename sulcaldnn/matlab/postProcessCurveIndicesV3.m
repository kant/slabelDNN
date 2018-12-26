function new_ids = postProcessCurveIndicesV3(spoint_id,surf_prefix,curve_prefix,p_thr,f_thr)
% f_thr - fill threshold
% p_thr = prune threshold

c=read_curve_sampling_descriptor2(surf_prefix, curve_prefix, false, 0);

% curve id
gid=c.geometry.conversion.v2c(spoint_id+1,1);

ugid=unique(gid)

spoint_id0=spoint_id;
for i=1:length(ugid)
    pred_ids=(spoint_id0(gid==ugid(i))'+1);
    curve_ids = (c.curve.base(ugid(i)).index);
    m=find(ismember(curve_ids,pred_ids));
    if ~(length(m) == length(curve_ids))
        % calculate length ratio between raw and label curve
	d1=calcCurveLength(curve_ids,c);

	d2=calcCurveLength(pred_ids,c);
        
        if (d2/d1 < p_thr) % distance ratio less than 10%
            spoint_id = setdiff(spoint_id, pred_ids-1);
        elseif (d2/d1 > f_thr) % distance ratio > 80% fill it
            spoint_id = setdiff(spoint_id, pred_ids-1);
            spoint_id = [spoint_id; curve_ids-1]
        else
            % Fill if beginning indices within 5mm range are missing
	    E_distance = calcCurveLength(curve_ids(1:m(1)),c);
            if E_distance < 5
                spoint_id = [spoint_id; curve_ids(1:m(1)-1)-1]
            end
            % Fill if ending indices within 5mm range are missing
	    E_distance = calcCurveLength(curve_ids(m(end):end),c);
            if E_distance < 5
                spoint_id = [spoint_id; curve_ids(m(end)+1:end)-1]
            end
            % loop through all the matching indices
            for i=1:length(m)-1
                if (m(i+1)-m(i)>1)
	    	    E_distance = calcCurveLength(curve_ids(m(i):m(i+1)),c);
                    if E_distance < 5
                        %cnt0=cnt0+1
                        spoint_id
                        spoint_id = [spoint_id; (curve_ids(m(i)+1:m(i+1)-1))-1]
                    else
                        %cnt1=cnt1+1
                        spoint_id = setdiff(spoint_id, curve_ids(m(i))-1);
                    end
                end
            end
        end
    end
end

new_ids=unique(spoint_id);

end
function dist = calcCurveLength(point,c)
    label = c.geometry.conversion.v2c(point);

    curves = unique(label);
    cnt=1;
    for i = curves'
        disp(i)
        cand = c.curve.base(i).index;
        cand = cand(ismember(cand, point(label == i)));
        vert0 = c.surface.vertex(cand,:);
        vert1 = [vert0(2:end,:); vert0(end,:)];
        dist = vert1 - vert0;
        dist = sqrt(sum(dist .* dist, 2));
        d(cnt)=sum(dist);
        cnt=cnt+1;
    end
    dist=sum(d);
end
