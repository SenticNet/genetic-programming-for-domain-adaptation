function [Tidxb,Yidxb]=newxy(Tidx,Yidx,nclass,rev_ind)

classp = []; sent_ind = [];
classt = []; k = 1;
cnt_classp = zeros(nclass,1);
cnt_classt = zeros(nclass,1);
rev_ind = rev_ind(end-size(Tidx)+1:end,:);
for i = 1:size(Tidx,1)
    if i==1
        sent_ind = rev_ind(i);        
    else
        if sent_ind ~= rev_ind(i) % same sentence      
          [a,b]=max(cnt_classt');
          Tidxb(k)=b;
          [a,b]=max(cnt_classp');
          Yidxb(k)=b;
          k=k+1;
          classt = []; classp = []; sent_ind = [];
          cnt_classp = zeros(nclass,1);
          cnt_classt = zeros(nclass,1);
          sent_ind = rev_ind(i);   
        end          
    end
          classt = Tidx(i);
          classp = Yidx(i);
          cnt_classp(classp)=cnt_classp(classp)+1;
          cnt_classt(classt)=cnt_classt(classt)+1;    
end

end