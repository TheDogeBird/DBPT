function sendMessage() {
  // get user input from input field
  var userInput = $('#user-input').val();
  // append user input to chat log
  $('#chat-log').append('<div class="chat-message chat-user"><p>' + userInput + '</p></div>');
  // clear input field
  $('#user-input').val('');
  // send user input to server for processing
  $.ajax({
    type: 'POST',
    url: '/get_response',
    data: {'msg': userInput},
    success: function(data) {
      // append response from server to chat log
      $('#chat-log').append(data);
      // scroll to bottom of chat log
      $('#chat-log').scrollTop($('#chat-log')[0].scrollHeight);
    },
    error: function() {
      // handle error
      console.log('Error processing message.');
    }
  });
}
