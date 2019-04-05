

open(FILE, $ARGV[0]);
while($line = <FILE>)
{
 @list = ();
 @list = split "\t",$line;
 if($list[0] eq "1.0"){$class = 0;}
 if($list[0] eq "2.0"){$class = 0;}
 if($list[0] eq "4.0"){$class = 1;}
 if($list[0] eq "5.0"){$class = 1;} 
 print "$class\n";
}
close(FILE);


