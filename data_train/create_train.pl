$dom1="books";
$dom2="dvd";
$dom3="music";

$lan1="en"; 
$lan2="de";
$lan3="fr"; 
$lan4="jp";

$src=$dom1;$trg=$dom3;$srcl=$lan1;$trgl=$lan3;
$srcpath = "/home/senticteam/dgpnn/exp2/cls-acl10-processed-tagged/$srcl/$src";
$trgpath = "/home/senticteam/dgpnn/exp2/cls-acl10-processed-tagged/$trgl/$trg";

system("cp $srcpath/test2_tagged.review source/train2_tagged.review");
system("cp $trgpath/test2_tagged.review target/train2_tagged.review");

$source = "$srcpath/train2.review";

open(FILE, $source);
$cnt = 0;
open(OUT, ">source/train2.review");
while($cnt<1000)
{
 $line = <FILE>;
 print OUT $line;
 $cnt++;
}
close(OUT);
close(FILE);

$target = "$trgpath/train2.review";

open(FILE, $target);
$cnt = 0;
open(OUT, ">target/train2.review");
while($cnt<1000)
{
 $line = <FILE>;
 print OUT $line;
 $cnt++;
}
close(OUT);
close(FILE);

$targetu = "$trgpath/unlabeled2.review";
open(FILE, $targetu);
$cnt = 0;
open(OUT, ">source/unlabeled2.review");
while($cnt<20000)
{
 $line = <FILE>;
 print OUT $line;
 $cnt++;
}
close(OUT);
close(FILE);

