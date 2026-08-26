/**
 * TẦNG 1 & 2 TRÌNH DUYỆT: BỘ ỔN ĐỊNH KÝ TỰ & BỘ ĐỆM CÂU (Phụ lục A - JavaScript ES Module)
 * Chạy trực tiếp trong trình duyệt sau MediaPipe Hands / ONNX Runtime Web.
 */

export class LetterStabilizer {
  constructor(opts = {}) {
    const o = Object.assign({
      window: 8,            // số khung hình trong cửa sổ bỏ phiếu M
      minVotes: 6,          // số phiếu tối thiểu N để chốt (N/M = 6/8 = 75%)
      minConf: 0.60,        // ngưỡng softmax
      releaseFrames: 4,     // phải "nhả" tay bao nhiêu khung hình mới chốt tiếp
      spaceFrames: 12,      // nghỉ tay >= 12 khung hình -> KHOẢNG TRẮNG
      holdRepeatFrames: 25, // giữ nguyên 1 ký tự lâu -> KÝ TỰ ĐÔI (LL, SS, EE)
      sentenceFrames: 45,   // nghỉ rất lâu -> KẾT THÚC CÂU
    }, opts);

    Object.assign(this, o);
    this.buf = [];
    this.state = 'IDLE';
    this.last = null;
    this.rel = 0;
    this.idle = 0;
    this.hold = 0;
    this.spaceEmitted = false;
    this.sentenceEmitted = false;
  }

  reset() {
    this.buf = [];
    this.state = 'IDLE';
    this.last = null;
    this.rel = 0;
    this.idle = 0;
    this.hold = 0;
    this.spaceEmitted = false;
    this.sentenceEmitted = false;
  }

  /**
   * @param {string|null} letter Ký tự dự đoán ('A'-'Z')
   * @param {number} conf Độ tin cậy (0.0 - 1.0)
   * @returns {null | {type: 'LETTER'|'SPACE'|'END', value?: string}}
   */
  push(letter, conf) {
    const low = letter == null || conf < this.minConf;

    if (low) {
      this.idle++;
    } else {
      this.idle = 0;
      this.spaceEmitted = false;
      this.sentenceEmitted = false;
    }

    // Phát hiện khoảng trắng (trễ 1 lần)
    if (this.idle >= this.spaceFrames && !this.spaceEmitted) {
      this.spaceEmitted = true;
      this.state = 'IDLE';
      this.last = null;
      this.buf = [];
      return { type: 'SPACE' };
    }

    // Phát hiện kết thúc câu
    if (this.idle >= this.sentenceFrames && !this.sentenceEmitted) {
      this.sentenceEmitted = true;
      this.reset();
      return { type: 'END' };
    }

    this.buf.push(low ? null : letter.toUpperCase());
    if (this.buf.length > this.window) {
      this.buf.shift();
    }

    const counts = new Map();
    for (const x of this.buf) {
      if (x) counts.set(x, (counts.get(x) || 0) + 1);
    }

    let top = null, votes = 0;
    for (const [k, v] of counts) {
      if (v > votes) {
        top = k;
        votes = v;
      }
    }

    const stable = top !== null && votes >= this.minVotes;

    if (this.state === 'RELEASE') {
      if (low || (stable && top !== this.last)) {
        if (++this.rel >= this.releaseFrames) {
          this.state = 'IDLE';
          this.rel = 0;
        }
      } else if (stable && top === this.last) {
        if (++this.hold >= this.holdRepeatFrames) {
          this.hold = 0;
          this.rel = 0;
          return { type: 'LETTER', value: this.last }; // Ký tự đôi: LL, SS, EE
        }
      }
      return null;
    }

    if (stable) {
      this.state = 'RELEASE';
      this.last = top;
      this.rel = 0;
      this.hold = 0;
      this.buf = [];
      return { type: 'LETTER', value: top };
    }

    return null;
  }
}

/**
 * Bộ đệm câu: Gom sự kiện từ stabilizer, hỗ trợ Optimistic UI và gọi API theo Debounce.
 */
export class SentenceBuffer {
  constructor({ onUpdate, correctFn, debounceMs = 700 } = {}) {
    this.words = [];
    this.cur = '';
    this.onUpdate = onUpdate || (() => {});
    this.correctFn = correctFn || (async ws => ws.join(' '));
    this.debounceMs = debounceMs;
    this._timer = null;
  }

  feed(ev) {
    if (!ev) return;

    if (ev.type === 'LETTER') {
      this.cur += ev.value;
    } else if (ev.type === 'SPACE') {
      if (this.cur) {
        this.words.push(this.cur);
        this.cur = '';
      }
    } else if (ev.type === 'END') {
      if (this.cur) {
        this.words.push(this.cur);
        this.cur = '';
      }
      this.flush();
      return;
    }

    // Hiển thị bản nháp Optimistic UI ngay lập tức
    this.onUpdate({
      raw: this.preview(),
      isFinal: false
    });

    this._debounceCorrect();
  }

  preview() {
    return [...this.words, this.cur].filter(Boolean).join(' ');
  }

  _debounceCorrect() {
    clearTimeout(this._timer);
    this._timer = setTimeout(() => this.flush(), this.debounceMs);
  }

  async flush() {
    clearTimeout(this._timer);
    const ws = [...this.words, this.cur].filter(Boolean);
    if (!ws.length) return;

    try {
      const result = await this.correctFn(ws);
      this.onUpdate({
        raw: ws.join(' '),
        english: typeof result === 'object' ? result.english : result,
        vietnamese: typeof result === 'object' ? result.translated : null,
        isFinal: true
      });
    } catch (err) {
      console.error('Error calling post-processing API:', err);
    }
  }

  clear() {
    clearTimeout(this._timer);
    this.words = [];
    this.cur = '';
    this.onUpdate({ raw: '', english: '', isFinal: true });
  }

  backspace() {
    if (this.cur) {
      this.cur = this.cur.slice(0, -1);
    } else if (this.words.length) {
      this.cur = this.words.pop().slice(0, -1);
    }
    this.onUpdate({ raw: this.preview(), isFinal: false });
  }
}
