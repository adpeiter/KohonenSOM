var cellsError;
var ids = [];
var values = [];
var interval = 4000;

cellsError = document.getElementsByClassName("error");

for (i = 0; i < cellsError.length; i++) {
	ids.push(cellsError[i].id);
	values.push(cellsError[i].innerText);
};

function startStopAlternateClass(start) {
	if (start)
		for (i = 0; i < ids.length; i++)
			alternateClass(i, 0);
	else if (start == undefined)
		for (i = 0; i < ids.length; i++)
			document.getElementById(ids[i]).className = "error";
}

function alternateClass(i, j) {
	if (j >= values[i].length)
		j = 0;
	var elem = document.getElementById(ids[i]);
	elem.className = "bg" + values[i][j] + " text-white";
	var ckb = document.getElementById("ckbChangeColorErrorNodes");
	if (ckb.checked)
		setTimeout(function () { alternateClass(i, ++j) }, interval);
	else
		startStopAlternateClass();
}