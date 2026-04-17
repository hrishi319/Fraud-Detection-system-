<script>
let streamInterval = null;

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const messageBox = document.getElementById("messageBox");

// START STREAM
startBtn.addEventListener("click", async () => {
  try {
    // Start backend stream
    const response = await fetch("/start_stream", {
      method: "POST"
    });

    const data = await response.json();

    // Show success message
    messageBox.textContent = data.message;
    messageBox.style.color = "#166534";
    messageBox.style.borderColor = "#86efac";
    messageBox.style.background = "#f0fdf4";

    // Prevent multiple intervals
    if (streamInterval) return;

    // Start live updates every 2 sec
    streamInterval = setInterval(() => {
      fetch("/get_transactions")
        .then(response => response.json())
        .then(data => {
          updateTable(data.transactions);
          updateStats(data.stats);
        })
        .catch(error => {
          console.error("Update error:", error);
        });
    }, 2000);

  } catch (error) {
    messageBox.textContent = "Error while starting stream";
    messageBox.style.color = "#b91c1c";
    messageBox.style.borderColor = "#fca5a5";
    messageBox.style.background = "#fef2f2";
  }
});

// STOP STREAM
stopBtn.addEventListener("click", async () => {
  try {
    // Stop backend stream
    const response = await fetch("/stop_stream", {
      method: "POST"
    });

    const data = await response.json();

    // Stop frontend live updates
    clearInterval(streamInterval);
    streamInterval = null;

    // Show stop message
    messageBox.textContent = data.message;
    messageBox.style.color = "#991b1b";
    messageBox.style.borderColor = "#fca5a5";
    messageBox.style.background = "#fef2f2";

  } catch (error) {
    messageBox.textContent = "Error while stopping stream";
    messageBox.style.color = "#b91c1c";
  }
});

// UPDATE TABLE
function updateTable(transactions) {
  const tbody = document.querySelector("tbody");
  tbody.innerHTML = "";

  transactions.forEach(txn => {
    tbody.innerHTML += `
      <tr>
        <td>${txn.id}</td>
        <td>${txn.user}</td>
        <td>₹${txn.amount}</td>
        <td>${txn.location}</td>
        <td>${txn.device}</td>
        <td>${txn.time}</td>
        <td><span class="badge">${txn.risk}</span></td>
        <td><span class="status">${txn.status}</span></td>
      </tr>
    `;
  });
}

// UPDATE STATS
function updateStats(stats) {
  document.querySelectorAll(".stat-card p")[0].innerText = stats.total;
  document.querySelectorAll(".stat-card p")[1].innerText = stats.flagged;
  document.querySelectorAll(".stat-card p")[2].innerText = stats.safe;
  document.querySelectorAll(".stat-card p")[3].innerText = stats.review;
}
</script>