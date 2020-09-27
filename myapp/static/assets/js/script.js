$('#drop a').click(function(){
    $(this).parent().find('input').click();
});

document.getElementById('upload').onchange = function() {
    document.getElementById('upload').submit();
};