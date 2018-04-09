       <?php 
        $upload_dir = "";
        $img = $_POST['hidden_data'];
        $name = $_POST['name'];
        $img = str_replace('data:image/png;base64,', '', $img);
        $img = str_replace(' ', '+', $img);
        $data = base64_decode($img);
        $file = $upload_dir . $name . ".jpg";
        $success = file_put_contents($file, $data);

        $uploadedfile = $file;
        $src = imagecreatefrompng($uploadedfile);
        list($width,$height)=getimagesize($uploadedfile);

        $newwidth1=32;
        $newheight1=32;
        $tmp1=imagecreatetruecolor($newwidth1,$newheight1);

        imagecopyresampled($tmp1,$src,0,0,0,0,$newwidth1,$newheight1,$width,$height);
        $filename1 = $name.".jpg". $_FILES['file']['name'];
        imagejpeg($tmp1,$filename1,100);

        imagedestroy($tmp1);

        print $success ? $file : 'Unable to save the file.';
        ?>


