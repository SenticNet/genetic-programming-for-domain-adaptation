load set_train_rnn0
load set_test_rnn0
load set_val_rnn0
dlmwrite('fold\train0_x',set_train_rnn0(1:end-1,:)');
dlmwrite('fold\train0_y',set_train_rnn0(end,:)'-1);
dlmwrite('fold\test0_y',set_test_rnn0(end,:)'-1);
dlmwrite('fold\test0_x',set_test_rnn0(1:end-1,:)');
dlmwrite('fold\val0_x',set_val_rnn0(1:end-1,:)');
dlmwrite('fold\val0_y',set_val_rnn0(end,:)'-1);