for($k=0;$k<=0;$k++){
system("rm ..\\neu_gp".$k."\/gp_pred\*");
system("rm ..\\neu_gp".$k."\/log_test.txt");
system("rm ..\\neu_gp".$k."\/log.txt");
system("rm ..\\neu_gp".$k."\/solution.txt");
print "java -cp \"lib\/\*\:class\/\:\.\" gp_nlp_train $k \n";
system("java -cp \"lib\/\*\:class\/\:\.\" gp_nlp_train $k");
}
