// ---- Active nav link ----
document.addEventListener("DOMContentLoaded", () => {
  const path = location.pathname.split("/").pop() || "index.html";
  document.querySelectorAll("nav.primary a").forEach(a => {
    if (a.getAttribute("href") === path) a.classList.add("active");
  });

  // ---- Scroll reveal ----
  const items = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window && items.length){
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting){ e.target.classList.add("in"); io.unobserve(e.target); }
      });
    }, { threshold: 0.12 });
    items.forEach(el => io.observe(el));
  } else {
    items.forEach(el => el.classList.add("in"));
  }

  // ---- "Issue" date stamp in masthead (edit or remove freely) ----
  const stamp = document.querySelector("[data-issue-date]");
  if (stamp){
    stamp.textContent = "Updated " + new Date().toLocaleDateString("en-US", { month: "long", year: "numeric" });
  }
});
