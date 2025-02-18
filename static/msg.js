$(document).ready(function () {
    $("#textForm").submit(function (e) {
        e.preventDefault();
        const userMessage = $("#text").val().trim();
        if (userMessage === "") return;

        $("#messageContainer").append(<div class="user-message">${userMessage}</div>);
        $("#text").val(""); 

        $("#messageContainer").append(<div class="typing-indicator"><span></span><span></span><span></span></div>);

        setTimeout(function () {
            $(".typing-indicator").remove(); 
            $.ajax({
                type: "POST",
                url: "/get_text_response",
                contentType: "application/json",
                data: JSON.stringify({ message: userMessage }),
                success: function (response) {
                    $("#messageContainer").append(<div class="bot-message">${response.response}</div>);
                },
                error: function () {
                    $("#messageContainer").append(<div class="bot-message">Error processing your request.</div>);
                }
            });
        }, 1000);
    });
});

function switchModel(modelKey) {
    $.ajax({
      url: "/switch_model",
      type: "POST",
      contentType: "application/json",
      data: JSON.stringify({ model: modelKey }),
      success: function(response) {
        alert(response.message);
      },
      error: function(err) {
        alert("Error: " + err.responseJSON.error);
      }
    });
  }
  
  function sendFeedback() {
    var question = prompt("Enter the question you asked:");
    var answer = prompt("Enter the answer provided by the chatbot:");
    var rating = prompt("Enter your rating (1-5):");
    var actual = prompt("Was the answer correct? Enter 1 for Yes, 0 for No:");
    $.ajax({
      url: "/feedback",
      type: "POST",
      contentType: "application/json",
      data: JSON.stringify({
        question: question,
        answer: answer,
        rating: rating,
        predicted: 1,
        actual: actual
      }),
      success: function(response) {
        alert(response.message);
      },
      error: function(err) {
        alert("Error: " + err.responseJSON.error);
      }
    });
  }
  
  function getEvaluation() {
    $.ajax({
      url: "/evaluation",
      type: "GET",
      success: function(response) {
        var msg = "Feedback Count: " + (response.feedback_count || 0) + "\n" +
                  "Accuracy: " + (response.accuracy || 0).toFixed(2) + "\n" +
                  "Precision: " + (response.precision || 0).toFixed(2) + "\n" +
                  "Recall: " + (response.recall || 0).toFixed(2) + "\n" +
                  "F1 Score: " + (response.f1_score || 0).toFixed(2);
        alert(msg);
      },
      error: function(err) {
        alert("Error retrieving evaluation metrics.");
      }
    });
  }