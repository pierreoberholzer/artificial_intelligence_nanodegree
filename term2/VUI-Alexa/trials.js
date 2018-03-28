var myNumbersArray = [1, 2, 3, 4, 5, 6, 7, 8];
var myStrArray = ["coucou", "caca", "coucou"];
var word = "coucou"
ind = myStrArray.filter(function(x){
    return x.includes(word);
});

console.log(ind);