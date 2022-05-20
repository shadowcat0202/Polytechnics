function addNumber(){
    var num1 = 2;
    var num2 = 3;
    var sum = num1 + num2;
    console.log("result = " + sum)
}
var x = 10;
function displayNumber(){
    console.log("x = " + x)
    console.log("y = " + y)
    var y = 20;
}

var test = function(a, b){
    return (a + b);
}
var input = prompt("입력받기 띄어쓰기로 구분하세요").split(' ');
console.log(parseInt(input[0]))
console.log(test(parseInt(input[0]), parseInt(input[1])))
