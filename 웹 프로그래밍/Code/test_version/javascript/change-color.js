var heading = document.querySelector('#heading1');
heading.onclick = function(){
    if (heading.style.color == "red"){
        heading.style.color = "black"
    }else{
        heading.style.color = "red"
    };
};