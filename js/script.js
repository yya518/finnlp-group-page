document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll('.box').forEach(box => {
    box.addEventListener('click', function () {
      const targetId = this.nextElementSibling.id;
      const targetSection = document.getElementById(targetId);
      const icon = this.querySelector('.expand-icon');
      const isCurrentlyOpen = targetSection.style.display === 'block';

      // Close all sections and reset icons
      document.querySelectorAll('.details-section').forEach(section => section.style.display = 'none');
      document.querySelectorAll('.expand-icon').forEach(i => {
        i.classList.remove('fa-chevron-up');
        i.classList.add('fa-chevron-down');
      });

      // Toggle clicked section
      if (!isCurrentlyOpen) {
        targetSection.style.display = 'block';
        icon.classList.remove('fa-chevron-down');
        icon.classList.add('fa-chevron-up');
      }
    });
  });
});