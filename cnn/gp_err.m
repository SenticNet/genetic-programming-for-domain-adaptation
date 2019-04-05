nclass = 2;
rev_ind = importdata('trainb_ind');

for k=1:10

filename = sprintf('gp/neu_gp/gp_pred%d',k);
gp_pred = importdata(filename);
filename = sprintf('gp/neu_gp/nlp_neu%d.txt',k);
data = importdata(filename);
gp_true = data(end,end-size(gp_pred)+1:end);

for i=1:size(gp_pred,1)
   if gp_pred(i)<1.5
       gp_pred2(i)=1;
   else
       gp_pred2(i)=2;
   end
end

[idxb,idx2b]=newxy(gp_pred2',gp_true',nclass,rev_ind);

cm = confusionmat(idxb,idx2b);

for x=1:nclass

tp = cm(x,x);
fp = sum(cm(:, x))-cm(x, x);
fn = sum(cm(x, :), 2)-cm(x, x);
pre(x)=tp/(tp+fp+0.01);
rec(x)=tp/(tp+fn+0.01);
fmea(x) = 2*pre(x)*rec(x)/(pre(x)+rec(x)+0.01);

end


classfmea(k,:)=fmea;
fmea = sum(fmea)/nclass;
allfmea(k)=fmea;

end
