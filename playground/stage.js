// Wires a `.stage` element to the simulation: lazy loading, status line, buttons.
const LABELS = {
  loading: 'waking the robot…',
  auto: 'playing by itself',
  locked: matchMedia('(hover: none)').matches ? 'playing by itself · tap a toy to drive' : 'playing by itself · click to drive',
  manual: 'you’re driving',
  error: 'the robot needs WebGL and WebAssembly',
};

export function initStage(stage, { autostart = true } = {}) {
  const status = stage.querySelector('.status');
  const wake = stage.querySelector('.stage-wake');
  const letPlay = stage.querySelector('[data-act="auto"]'); // only offered while the visitor is driving
  const setState = (state) => {
    status.dataset.state = state;
    status.textContent = LABELS[state];
    if (letPlay && state !== 'loading') letPlay.hidden = state !== 'manual';
  };
  let started = false;
  const start = async () => {
    if (started) return;
    started = true;
    wake?.remove();
    setState('loading');
    try {
      const { mount } = await import('./view.js?v=2');
      const view = await mount(stage, { onState: setState });
      stage.querySelectorAll('[data-act]').forEach((button) => {
        if (button !== letPlay) button.hidden = false;
        button.addEventListener('click', () => view[button.dataset.act]());
      });
      window.playground = view; // handy in the console
    } catch (error) {
      console.error(error);
      setState('error');
    }
  };

  // Phones on a data budget and people who asked for less motion get a button instead.
  const frugal = navigator.connection?.saveData || matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (autostart && !frugal) {
    const begin = () => (window.requestIdleCallback ?? setTimeout)(start);
    if (document.readyState === 'complete') begin();
    else addEventListener('load', begin, { once: true });
  } else {
    wake.hidden = false;
    wake.addEventListener('click', start);
    status.textContent = '';
  }
}
