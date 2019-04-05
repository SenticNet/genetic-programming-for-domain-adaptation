$p=$ARGV[1];
system("rm ..\/neu_gp".$p."\/g\p_pred\*");

open(FILE, $ARGV[0]);
while($line = <FILE>)
{
 chomp($line);
 if(index($line,"start")>=0){
 $neu=substr($line,index($line,":")+1);
 }
 else{
  $labelp=substr($line,index($line,":")+1);
  if($labelp eq "NaN"){$labelp = 1;}
  open(FILE2,">>..\/neu_gp".$p."\/gp_pred".$neu);
  print FILE2 $labelp."\n";
  close(FILE2);
 }
}
close(FILE);
