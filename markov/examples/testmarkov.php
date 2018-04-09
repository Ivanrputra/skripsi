<?php
	$var1 = 'classify';
	$var2 = '1.jpg'; 	
	exec("python C:/xampp/htdocs/py/train.py $var1 $var2",$output);
	var_dump($output);
?>