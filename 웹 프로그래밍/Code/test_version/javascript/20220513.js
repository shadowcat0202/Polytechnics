var p2 = document.querySelector("#p2");
p2.onclick = function () {
    h2 = document.querySelector("#h2")
    if (h2.style.color != 'red') 
        h2.style.color = 'red';
    else 
        h2.style.color = 'blue';
    }

function mul() {
    // alert("mul() in");
    var input1 = document.getElementById("input_num1").value;
    var input2 = document.getElementById("input_num2").value;
    // var array_size = 10;
    // var array = new Array(array_size)
    // for (let i = 0; i < array_size; i++) {
    //     array[i] = (i + 1) * input1;
    //     // document.write("<p>"+array[i]+"</p>");   //이 방법은 완전 html을 새로 작성한다
    //     // 기존 열려있는 html창에 출력하고 싶다면 밑에다가 출력창을 미리 만들어 놓고 그 위치에 뿌려주는 방법으로 한다        
    // }
    // document.getElementById("mul_result").value = array;
    // alert(input1 + "의 배수는:" + array + "\n두 수의 곱셈은" + (input1 * input2));
    alert(input1)
    var star = "";
    for(let i = 0; i < input1; i++){
        var line = "";
        for(let b =input1-i-1;  b > 0; b--){
            // line.concat(" ");
            document.write(" ");
        }
        for(let j = 0; j < i+1; j++){
            document.write("*");
            // line.concat("*");
        }
        // line.concat("*");
        // alert(line);
        document.write("<br>");
    }
    // document.write(output);
}
