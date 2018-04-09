<html>

    <script src="http://ajax.googleapis.com/ajax/libs/jquery/1.11.0/jquery.min.js" ></script>
    <script type="text/javascript">
    var canvas, ctx, flag = false,
        prevX = 0,
        a=0,
        b=0,
        currX = 0,
        prevY = 0,
        currY = 0,
        dot_flag = false;

    var x = "black",
        y = 10;

    function reset(){
        canvas = document.getElementById('can');
        ctx = canvas.getContext("2d");
        
        w = canvas.width;
        h = canvas.height;
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        // hdc.fillStyle = "#9ea7b8";
        // hdc.fillRect(0,0,w,h);
    
        canvas.addEventListener("mousemove", function (e) {
            findxy('move', e)
        }, false);
        canvas.addEventListener("mousedown", function (e) {
            findxy('down', e)
        }, false);
        canvas.addEventListener("mouseup", function (e) {
            findxy('up', e)
        }, false);
        canvas.addEventListener("mouseout", function (e) {
            findxy('out', e)
        }, false);
    }
    
    function init() {
        canvas = document.getElementById('can');
        ctx = canvas.getContext("2d");
        w = canvas.width;
        h = canvas.height;
    
        canvas.addEventListener("mousemove", function (e) {
            findxy('move', e)
        }, false);
        canvas.addEventListener("mousedown", function (e) {
            findxy('down', e)
        }, false);
        canvas.addEventListener("mouseup", function (e) {
            findxy('up', e)
        }, false);
        canvas.addEventListener("mouseout", function (e) {
            findxy('out', e)
        }, false);
    }
    function color(obj) {
        switch (obj.id) {
            case "green":
                x = "green";
                break;
            case "blue":
                x = "blue";
                break;
            case "red":
                x = "red";
                break;
            case "yellow":
                x = "yellow";
                break;
            case "orange":
                x = "orange";
                break;
            case "black":
                x = "black";
                break;
            case "white":
                x = "white";
                break;
            case "purple":
                x = "purple";
                break;
        }
        if (x == "white") y = 14;
        else y = 10;
    }
    function draw() {
        ctx.beginPath();
        ctx.moveTo(prevX, prevY);
        ctx.lineTo(currX, currY);
        ctx.strokeStyle = x;
        ctx.lineWidth = y;
        ctx.stroke();
        ctx.closePath();
    }
    function erase() {
        var m = confirm("Want to clear");
        if (m) {
            ctx.clearRect(0, 0, w, h);
            document.getElementById("canvasimg").style.display = "none";
        }
    }
    function save() {

        canvas = document.getElementById("can");
        var dataURL = canvas.toDataURL("image/png");
        document.getElementById('hidden_data').value = dataURL;
        var fd = new FormData(document.forms["form1"]);

        var xhr = new XMLHttpRequest();
        xhr.open('POST', 'upload_data.php', true);
        a = a+1;
        xhr.upload.onprogress = function(e) {
            if (e.lengthComputable) {
                var percentComplete = (e.loaded / e.total) * 100;
                console.log(percentComplete + '% uploaded');
                alert('Succesfully uploaded');
                print
            }
        };

        xhr.onload = function() {

        };
        xhr.send(fd);
    };
  
        // document.getElementById("canvasimg").style.border = "2px solid";
        // var dataURL = canvas.toDataURL();
        // document.getElementById("canvasimg").src = dataURL;
        // document.getElementById("canvasimg").style.display = "inline";
        // var canvas = document.getElementById("can");
        
        // var dataURL = canvas.toDataURL("image/png");

        // document.getElementById('my_hidden').value = canvas.toDataURL('image/png');
        // document.forms["form1"].submit();

        // // var img    = canvas.toDataURL("image/png")

        // var MIME_TYPE = "image/png";
        // var imgURL = canvas.toDataURL(MIME_TYPE);

        // var dlLink = document.createElement('a');
        // dlLink.download = "images/aa";
        // dlLink.href = imgURL;
        // dlLink.dataset.downloadurl = [MIME_TYPE, dlLink.download, dlLink.href].join(':');

        // document.body.appendChild(dlLink);
        // dlLink.click();
        // document.body.removeChild(dlLink);
        // document.write('<img src="'+img+'"/>');
    // }
    
    function findxy(res, e) {
        if (res == 'down') {
            prevX = currX;
            prevY = currY;
            currX = e.clientX - canvas.offsetLeft;
            currY = e.clientY - canvas.offsetTop;
    
            flag = true;
            dot_flag = true;
            if (dot_flag) {
                ctx.beginPath();
                ctx.fillStyle = x;
                ctx.fillRect(currX, currY, 2, 2);
                ctx.closePath();
                dot_flag = false;
            }
        }
        if (res == 'up' || res == "out") {
            flag = false;
        }
        if (res == 'move') {
            if (flag) {
                prevX = currX;
                prevY = currY;
                currX = e.clientX - canvas.offsetLeft;
                currY = e.clientY - canvas.offsetTop;
                draw();
            }
        }
    }
    </script>
    <style>
        table, th, td {
            border: 1px solid black;
        }
        div { margin: auto; }
    </style>

    <body>
        
    <script type="text/javascript">
    window.onload = function() {
        canvas = document.getElementById('can');
        ctx = canvas.getContext("2d");
        
        w = canvas.width;
        h = canvas.height;
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        // hdc.fillStyle = "#9ea7b8";
        // hdc.fillRect(0,0,w,h);
    
        canvas.addEventListener("mousemove", function (e) {
            findxy('move', e)
        }, false);
        canvas.addEventListener("mousedown", function (e) {
            findxy('down', e)
        }, false);
        canvas.addEventListener("mouseup", function (e) {
            findxy('up', e)
        }, false);
        canvas.addEventListener("mouseout", function (e) {
            findxy('out', e)
        }, false);
    }
    </script>
        <p id="demo"></p>
        
            <table style="width:500px;margin:auto;">
                <tr>
                    
                    <th></th>
                    <th></th>
                    <th></th>
                    <th></th>
                    <th></th>
                </tr>
                <tr>
                    <td rowspan="8" colspan="3" width="400" height="400"><canvas id="can" width="400" height="400" style="margin:auto;border:2px solid;"></canvas></td>
                    <td><div >Choose Color</div></td>
                    <td><div >Erase</div></td>
                </tr>
                <tr>
                    <td><div style="width:10px;height:10px;background:green;border:2px solid" id="green" onclick="color(this)"></div></td>
                    <td><div style="width:15px;height:15px;background:white;border:2px solid;" id="white" onclick="color(this)"></div></td>
                </tr>
                <tr>
                    <td><div style="width:10px;height:10px;background:red;border:2px solid" id="red" onclick="color(this)"></div></td>
                    <td> </div></td>
                </tr>
                <tr>
                    <td><div style="width:10px;height:10px;background:blue;border:2px solid" id="blue" onclick="color(this)"></div></td>
                    <td> </div></td>
                </tr>
                <tr>
                    <td><div style="width:10px;height:10px;background:purple;border:2px solid;" id="purple" onclick="color(this)"></div></td>
                    <td> </div></td>
                </tr>
                <tr>
                    <td><div style="width:10px;height:10px;background:yellow;border:2px solid" id="yellow" onclick="color(this)"></div></td>
                    <td> </div></td>
                </tr>
                <tr>
                    <td><div style="width:10px;height:10px;background:orange;border:2px solid" id="orange" onclick="color(this)"></div></td>
                    <td> </div></td>
                </tr>
                <tr>
                    <td><div style="width:10px;height:10px;background:black;border:2px solid" id="black" onclick="color(this)"></div></td>
                    <td> </div></td>
                </tr>
                <tr>
                    <td style="text-align:center;"><input  type="button" value="Simpan" id="btn" size="30" onclick="save()" ></td>
                    <td style="text-align:center;"><input type="button" value="Reset" id="clr" size="23" onclick="reset()" ></td>
                    <td style="text-align:center;"><a href='index.php?Class=true'><input type="button" value="Class" id="Class" size="23"></a></td>
                    <td></td>
                    <td></td>
                </tr>
            </table>
                <img id="canvasimg" style="position:absolute;top:10%;left:52%;" style="display:none;">
                
                
                    
            
            
            <br>
        <!-- <a href='index.php?train=true'><input type="button" value="Train" id="train" size="23"  style="position:absolute;top:55%;left:25%;"></a> -->
        
        <form method="post" accept-charset="utf-8" name="form1">
            <input name="hidden_data" id='hidden_data' type="hidden"/>
            <input type="hidden" name="name" id="name" value="1">
        </form>
        <?php
            if (isset($_GET['Class'])) {
                $var1 = 'classify';
                $var2 = array('1.jpg'); 
                for ($i=0; $i <count($var2) ; $i++) { 
                    $output = null;
                    $output1 = null;
                    $var3 = $var2[$i];
                    exec("python C:/xampp/htdocs/py/train.py $var1 $var3",$output,$output1);
                    // print_r($output);
                    // var_dump($output);
                    echo $output[0];    
                    // echo $output1;
                    
                    // $aa = 'aku';
                    // exec("python C:/xampp/htdocs/py/basic.py $aa ",$output1);
                    // // var_dump($output1);
                    // echo $output1[3];    
                    // // $output1 = exec("python C:/xampp/htdocs/py/markov/examples/basic.py",$output1);
                    // // var_dump($output1);
                }
            }
            // else if (isset($_GET['train'])) {
            //  $var1 = 'train';
            //  $var2 = array('1.jpg'); 
            //  for ($i=0; $i <count($var2) ; $i++) { 
            //      $output = null;
            //      $output1 = null;
            //      $var3 = $var2[$i];
            //      exec("python C:/xampp/htdocs/py/train.py $var1 $var3",$output);
            //      var_dump($output);
            //      // echo $output[3]; 
                    
            //      // $aa = 'aku';
            //      // exec("python C:/xampp/htdocs/py/basic.py $aa ",$output1);
            //      // // var_dump($output1);
            //      // echo $output1[3];    
            //      // // $output1 = exec("python C:/xampp/htdocs/py/markov/examples/basic.py",$output1);
            //      // // var_dump($output1);
            //  }
            // }

        ?>
        
    </body>
    </html>
