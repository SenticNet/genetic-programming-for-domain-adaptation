nclass = 2;
neu = 1;
cmall = zeros(2,2);
for p=0:0

allfmea = zeros(1,neu);
%p =0;

for k=1:neu

filename = sprintf('../neu_gp%d/gp_pred%d',p,k);
if exist(filename, 'file') == 2

gp_pred = importdata(filename);
gp_pred2 = [];
filename = sprintf('../neu_gp%d/nlp_neu%d.txt',p,k);
data = importdata(filename);
gp_true = data(end,end-size(gp_pred,1)+1:end);

%{
for i=1:size(gp_pred,1)
   if gp_pred(i)<1.5
       gp_pred2(i)=1;
   elseif gp_pred(i)<2.5
       gp_pred2(i)=2;
   elseif gp_pred(i)<3.5
       gp_pred2(i)=3;
   elseif gp_pred(i)>=3.5
       gp_pred2(i)=4;
   end
end
%}

for i=1:size(gp_pred,1)
   if gp_pred(i)<1.5
       gp_pred2(i)=1;
   else
       gp_pred2(i)=2;
   end
end

% 
% if size(gp_pred2,2)>15
%     endp=15;
% else
%     endp=size(gp_pred2,2)-2;
% end
%[idxb,idx2b]=newxy(gp_pred2',gp_true',nclass,rev_ind);
%cm = confusionmat(gp_pred2(1:10)',gp_true(1:10)');
%cm = confusionmat(gp_pred2(1:endp)',gp_true(1:endp)');
cm = confusionmat(gp_pred2(1:end)',gp_true(1:end)');

if size(cm,1)==1
    cm2 = [0,0;0,0];
    cm2(1,1)=cm;
    cm = cm2;
end
for x=1:nclass

tp = cm(x,x);

tn = cm(1,1);
for y=2:nclass
tn = tn+cm(y,y);
end
tn = tn-cm(x,x);


fp = sum(cm(:, x))-cm(x, x);
fn = sum(cm(x, :), 2)-cm(x, x);
pre(x)=tp/(tp+fp+0.01);
rec(x)=tp/(tp+fn+0.01);
%fmea(x) = 2*pre(x)*rec(x)/(pre(x)+rec(x)+0.01);
fmea(x) = (tp+tn)/(tp+fp+tn+fn);

end


classfmea(k,:)=fmea;
fmea = sum(fmea)/nclass;
allfmea(k)=fmea;
cmall = cmall+cm;
end

end

filename = sprintf('../neu_gp%d/test_ind%d',p,p);
test_ind = importdata(filename);
%z = [allfmea;test_ind']

cntk=0;
for k1=1:neu
   if allfmea(k1)==0
       cntk=cntk+1;
   end
end

fmea = sum(allfmea)/(neu-cntk)

fmeaall(p+1)=fmea;

maxacc = [];k2=1;
for k1=1:neu
  if allfmea(k1)<0.7
      maxacc(k2)=k1;
      k2 = k2 + 1;
  end
end

filename = sprintf('maxacc%d.txt',p);
dlmwrite(filename,maxacc);

end

dlmwrite('allfmeafinal.txt',fmea);
exit
