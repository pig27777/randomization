(() => {
  document.querySelectorAll('[data-carousel]').forEach((carousel) => {
    const slides = Array.from(carousel.querySelectorAll('[data-slide]'));
    const counter = carousel.querySelector('[data-counter]');
    const previous = carousel.querySelector('[data-previous]');
    const next = carousel.querySelector('[data-next]');
    let index = 0;

    function show(nextIndex) {
      const oldVideo = slides[index].querySelector('video');
      if (oldVideo) oldVideo.pause();
      index = (nextIndex + slides.length) % slides.length;
      slides.forEach((slide, i) => {
        slide.hidden = i !== index;
        slide.setAttribute('aria-label', `${i + 1} of ${slides.length}`);
      });
      counter.textContent = `${String(index + 1).padStart(2, '0')} / ${String(slides.length).padStart(2, '0')}`;
      const video = slides[index].querySelector('video');
      if (video && video.preload === 'none') {
        video.preload = 'metadata';
        video.load();
      }
    }

    previous.addEventListener('click', () => show(index - 1));
    next.addEventListener('click', () => show(index + 1));
    carousel.addEventListener('keydown', (event) => {
      if (event.target.closest('video')) return;
      if (event.key === 'ArrowLeft') {
        event.preventDefault();
        show(index - 1);
      } else if (event.key === 'ArrowRight') {
        event.preventDefault();
        show(index + 1);
      }
    });
    show(0);
  });

  document.querySelectorAll('[data-model-explorer]').forEach((explorer) => {
    const tabs = Array.from(explorer.querySelectorAll('[role="tab"]'));
    const panels = Array.from(explorer.querySelectorAll('[role="tabpanel"]'));

    function select(index) {
      tabs.forEach((tab, i) => {
        tab.setAttribute('aria-selected', String(i === index));
        tab.tabIndex = i === index ? 0 : -1;
        panels[i].hidden = i !== index;
      });
    }

    tabs.forEach((tab, index) => {
      tab.addEventListener('click', () => select(index));
      tab.addEventListener('keydown', (event) => {
        let target;
        if (event.key === 'ArrowRight') target = (index + 1) % tabs.length;
        else if (event.key === 'ArrowLeft') target = (index + tabs.length - 1) % tabs.length;
        else if (event.key === 'Home') target = 0;
        else if (event.key === 'End') target = tabs.length - 1;
        if (target === undefined) return;
        event.preventDefault();
        select(target);
        tabs[target].focus();
      });
    });
  });
})();
