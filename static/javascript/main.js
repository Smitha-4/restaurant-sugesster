var divElement = document.getElementById('viz1706870423209');
var vizElement = divElement.getElementsByTagName('object')[0];
if (divElement.offsetWidth > 800) {
    vizElement.style.width = '1500px'; vizElement.style.height = '1527px';
} else if (divElement.offsetWidth > 500) {
    vizElement.style.width = '1500px'; vizElement.style.height = '1527px';
} else {
    vizElement.style.width = '100%'; vizElement.style.height = '3477px';
}
var scriptElement = document.createElement('script');
scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
vizElement.parentNode.insertBefore(scriptElement, vizElement);  