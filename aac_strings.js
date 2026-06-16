/**
 * aac_strings.js — AfriChess AAC announcement strings
 * All text spoken aloud by the voice accessibility engine.
 *
 * Adding a language: add a key to AAC_STRINGS matching the
 * BCP-47 language tag used in the language picker.
 * Chess piece names must match what parseSpokenMove() expects.
 */

const AAC_STRINGS = {

  // ── English ─────────────────────────────────────────────────────────────────
  'en': {
    lang:         'English',
    speechCode:   'en-US',          // passed to SpeechSynthesis and SpeechRecognition
    altCode:      'en-GB',          // fallback if primary voice not available

    // Piece names — used in move announcements AND in voice input parsing
    pieces: {
      p: 'Pawn',
      n: 'Knight',
      b: 'Bishop',
      r: 'Rook',
      q: 'Queen',
      k: 'King',
    },

    // Move announcements
    moves: {
      moveTo:       (piece, sq)        => `${piece} to ${sq}`,
      captures:     (piece, sq)        => `${piece} takes ${sq}`,
      castle_k:     ()                 => 'Castles kingside',
      castle_q:     ()                 => 'Castles queenside',
      promotes:     (piece, sq, promo) => `${piece} promotes to ${promo} on ${sq}`,
      check:        ()                 => 'Check',
      checkmate:    (winner)           => `Checkmate — ${winner} wins`,
      stalemate:    ()                 => 'Stalemate — the game is drawn',
      draw:         ()                 => 'The game is drawn',
    },

    // Game state announcements
    game: {
      yourTurn:     ()                 => 'Your turn',
      opponentTurn: (name)            => `${name}'s turn`,
      youPlay:      (color)           => `You play as ${color}`,
      white:        'White',
      black:        'Black',
      timeLeft:     (t)               => `${t} remaining`,
      gameStart:    ()                 => 'Game started',
      youWin:       ()                 => 'You win',
      youLose:      ()                 => 'You lose',
      draw:         ()                 => 'Draw',
    },

    // Panel UI labels
    ui: {
      toggle:       'Voice Chess',
      language:     'Language',
      listening:    'Listening...',
      notHeard:     'Could not hear move — please try again',
      illegal:      (move)            => `${move} is not a legal move`,
      confirm:      (move)            => `Did you mean ${move}? Say yes or no`,
      voiceOn:      'Voice announcements on',
      voiceOff:     'Voice announcements off',
      micOn:        'Microphone on — speak your move',
      micOff:       'Microphone off',
      gridTitle:    'Select piece',
      gridSquare:   'Select square',
      scanning:     'Switch scanning on',
    },

    // Voice input — words that map to pieces (for parser)
    voicePieces: {
      'pawn': 'p', 'knight': 'n', 'bishop': 'b',
      'rook': 'r', 'queen': 'q', 'king': 'k',
    },
    voiceFiles:   { 'alpha':'a','bravo':'b','charlie':'c','delta':'d','echo':'e','foxtrot':'f','golf':'g','hotel':'h' },
    voiceYes:     ['yes', 'yeah', 'correct', 'right', 'yep', 'confirm'],
    voiceNo:      ['no', 'nope', 'wrong', 'cancel', 'incorrect'],
    voiceCastle:  ['castle', 'castles', 'castling', 'short castle', 'kingside'],
    voiceCastleQ: ['castle queenside', 'long castle', 'queenside'],
  },

  // ── Swahili ─────────────────────────────────────────────────────────────────
  'sw': {
    lang:         'Kiswahili',
    speechCode:   'sw-KE',
    altCode:      'sw-TZ',

    pieces: {
      p: 'Askari',       // pawn — soldier
      n: 'Farasi',       // knight — horse
      b: 'Askofu',       // bishop
      r: 'Ngome',        // rook — castle/fort
      q: 'Malkia',       // queen
      k: 'Mfalme',       // king
    },

    moves: {
      moveTo:       (piece, sq)        => `${piece} kwenda ${sq}`,
      captures:     (piece, sq)        => `${piece} anachukua ${sq}`,
      castle_k:     ()                 => 'Ngome upande wa mfalme',
      castle_q:     ()                 => 'Ngome upande wa malkia',
      promotes:     (piece, sq, promo) => `${piece} anakuwa ${promo} kwenye ${sq}`,
      check:        ()                 => 'Shaka',
      checkmate:    (winner)           => `Shaka maarufu — ${winner} ameshinda`,
      stalemate:    ()                 => 'Mchezo umefungwa — sare',
      draw:         ()                 => 'Mchezo wa sare',
    },

    game: {
      yourTurn:     ()                 => 'Zamu yako',
      opponentTurn: (name)            => `Zamu ya ${name}`,
      youPlay:      (color)           => `Unacheza kama ${color}`,
      white:        'Nyeupe',
      black:        'Nyeusi',
      timeLeft:     (t)               => `${t} zimebaki`,
      gameStart:    ()                 => 'Mchezo umeanza',
      youWin:       ()                 => 'Umeshinda',
      youLose:      ()                 => 'Umeshindwa',
      draw:         ()                 => 'Sare',
    },

    ui: {
      toggle:       'Sauti ya Chess',
      language:     'Lugha',
      listening:    'Sikiliza...',
      notHeard:     'Sikusikia hatua — jaribu tena',
      illegal:      (move)            => `${move} si hatua halali`,
      confirm:      (move)            => `Unamaanisha ${move}? Sema ndiyo au hapana`,
      voiceOn:      'Matangazo ya sauti yamewashwa',
      voiceOff:     'Matangazo ya sauti yamezimwa',
      micOn:        'Kipaza sauti kimewashwa — sema hatua yako',
      micOff:       'Kipaza sauti kimezimwa',
      gridTitle:    'Chagua kipande',
      gridSquare:   'Chagua mraba',
      scanning:     'Uchunguzi wa swichi umewashwa',
    },

    voicePieces: {
      'askari': 'p', 'farasi': 'n', 'askofu': 'b',
      'ngome': 'r', 'malkia': 'q', 'mfalme': 'k',
    },
    voiceFiles:   { 'alpha':'a','bravo':'b','charlie':'c','delta':'d','echo':'e','foxtrot':'f','golf':'g','hotel':'h' },
    voiceYes:     ['ndiyo', 'ndio', 'sawa'],
    voiceNo:      ['hapana', 'la', 'siyo'],
    voiceCastle:  ['ngome mfalme', 'ngome fupi'],
    voiceCastleQ: ['ngome malkia', 'ngome ndefu'],
  },

  // ── French ──────────────────────────────────────────────────────────────────
  'fr': {
    lang:         'Français',
    speechCode:   'fr-FR',
    altCode:      'fr-BE',

    pieces: {
      p: 'Pion',
      n: 'Cavalier',
      b: 'Fou',
      r: 'Tour',
      q: 'Dame',
      k: 'Roi',
    },

    moves: {
      moveTo:       (piece, sq)        => `${piece} en ${sq}`,
      captures:     (piece, sq)        => `${piece} prend en ${sq}`,
      castle_k:     ()                 => 'Petit roque',
      castle_q:     ()                 => 'Grand roque',
      promotes:     (piece, sq, promo) => `${piece} se promeut en ${promo} en ${sq}`,
      check:        ()                 => 'Échec',
      checkmate:    (winner)           => `Échec et mat — ${winner} gagne`,
      stalemate:    ()                 => 'Pat — partie nulle',
      draw:         ()                 => 'Partie nulle',
    },

    game: {
      yourTurn:     ()                 => 'À vous de jouer',
      opponentTurn: (name)            => `Au tour de ${name}`,
      youPlay:      (color)           => `Vous jouez les ${color}`,
      white:        'Blancs',
      black:        'Noirs',
      timeLeft:     (t)               => `${t} restant`,
      gameStart:    ()                 => 'La partie commence',
      youWin:       ()                 => 'Vous gagnez',
      youLose:      ()                 => 'Vous perdez',
      draw:         ()                 => 'Nulle',
    },

    ui: {
      toggle:       'Échecs vocaux',
      language:     'Langue',
      listening:    'Écoute en cours...',
      notHeard:     'Coup non reconnu — veuillez réessayer',
      illegal:      (move)            => `${move} est un coup illégal`,
      confirm:      (move)            => `Voulez-vous dire ${move} ? Dites oui ou non`,
      voiceOn:      'Annonces vocales activées',
      voiceOff:     'Annonces vocales désactivées',
      micOn:        'Microphone activé — dites votre coup',
      micOff:       'Microphone désactivé',
      gridTitle:    'Choisir la pièce',
      gridSquare:   'Choisir la case',
      scanning:     'Balayage par contacteur activé',
    },

    voicePieces: {
      'pion': 'p', 'cavalier': 'n', 'fou': 'b',
      'tour': 'r', 'dame': 'q', 'roi': 'k',
    },
    voiceFiles:   { 'alpha':'a','bravo':'b','charlie':'c','delta':'d','echo':'e','foxtrot':'f','golf':'g','hotel':'h' },
    voiceYes:     ['oui', 'ouais', 'correct', 'confirmer'],
    voiceNo:      ['non', 'annuler', 'incorrect'],
    voiceCastle:  ['petit roque', 'roque court'],
    voiceCastleQ: ['grand roque', 'roque long'],
  },

  // ── Arabic ──────────────────────────────────────────────────────────────────
  'ar': {
    lang:         'العربية',
    speechCode:   'ar-SA',
    altCode:      'ar-EG',

    pieces: {
      p: 'جندي',         // soldier/pawn
      n: 'حصان',         // horse/knight
      b: 'فيل',          // elephant/bishop
      r: 'قلعة',         // castle/rook
      q: 'وزير',         // vizier/queen
      k: 'ملك',          // king
    },

    moves: {
      moveTo:       (piece, sq)        => `${piece} إلى ${sq}`,
      captures:     (piece, sq)        => `${piece} يأخذ ${sq}`,
      castle_k:     ()                 => 'تبييت قصير',
      castle_q:     ()                 => 'تبييت طويل',
      promotes:     (piece, sq, promo) => `${piece} يُرقَّى إلى ${promo} في ${sq}`,
      check:        ()                 => 'كِش',
      checkmate:    (winner)           => `كِش مات — ${winner} يفوز`,
      stalemate:    ()                 => 'تعادل بالإيقاف',
      draw:         ()                 => 'تعادل',
    },

    game: {
      yourTurn:     ()                 => 'دورك',
      opponentTurn: (name)            => `دور ${name}`,
      youPlay:      (color)           => `تلعب بـ${color}`,
      white:        'الأبيض',
      black:        'الأسود',
      timeLeft:     (t)               => `${t} متبقية`,
      gameStart:    ()                 => 'بدأت اللعبة',
      youWin:       ()                 => 'أنت تفوز',
      youLose:      ()                 => 'أنت تخسر',
      draw:         ()                 => 'تعادل',
    },

    ui: {
      toggle:       'الشطرنج الصوتي',
      language:     'اللغة',
      listening:    'جارٍ الاستماع...',
      notHeard:     'لم يُسمع الحركة — حاول مرة أخرى',
      illegal:      (move)            => `${move} حركة غير قانونية`,
      confirm:      (move)            => `هل تقصد ${move}؟ قل نعم أو لا`,
      voiceOn:      'الإعلانات الصوتية مفعّلة',
      voiceOff:     'الإعلانات الصوتية مُعطَّلة',
      micOn:        'الميكروفون مفعّل — قل حركتك',
      micOff:       'الميكروفون مُعطَّل',
      gridTitle:    'اختر القطعة',
      gridSquare:   'اختر المربع',
      scanning:     'المسح بالمفتاح مفعّل',
    },

    voicePieces: {
      'جندي': 'p', 'حصان': 'n', 'فيل': 'b',
      'قلعة': 'r', 'وزير': 'q', 'ملك': 'k',
    },
    voiceFiles:   { 'alpha':'a','bravo':'b','charlie':'c','delta':'d','echo':'e','foxtrot':'f','golf':'g','hotel':'h' },
    voiceYes:     ['نعم', 'أيوه', 'صحيح'],
    voiceNo:      ['لا', 'خطأ', 'إلغاء'],
    voiceCastle:  ['تبييت قصير'],
    voiceCastleQ: ['تبييت طويل'],
  },

  // ── Yoruba ──────────────────────────────────────────────────────────────────
  'yo': {
    lang:         'Yorùbá',
    speechCode:   'yo-NG',
    altCode:      'en-NG',          // fallback to Nigerian English if Yoruba not available

    pieces: {
      p: 'Ọmọ-ogun',    // soldier
      n: 'Ẹṣin',         // horse
      b: 'Bíṣọ́ọ̀bù',    // bishop (loanword)
      r: 'Ilé-olódi',   // fortress
      q: 'Ayaba',        // queen
      k: 'Ọba',          // king
    },

    moves: {
      moveTo:       (piece, sq)        => `${piece} lọ sí ${sq}`,
      captures:     (piece, sq)        => `${piece} gba ${sq}`,
      castle_k:     ()                 => 'Ilé-olódi ọba',
      castle_q:     ()                 => 'Ilé-olódi ayaba',
      promotes:     (piece, sq, promo) => `${piece} di ${promo} ní ${sq}`,
      check:        ()                 => 'Ọba wà nínú ewu',
      checkmate:    (winner)           => `Checkmate — ${winner} borí`,
      stalemate:    ()                 => 'Dídópin pẹ̀lú àdéhùn',
      draw:         ()                 => 'Àdéhùn',
    },

    game: {
      yourTurn:     ()                 => 'Àkókò rẹ ni',
      opponentTurn: (name)            => `Àkókò ${name} ni`,
      youPlay:      (color)           => `O ń ṣeré pẹ̀lú ${color}`,
      white:        'Funfun',
      black:        'Dúdú',
      timeLeft:     (t)               => `${t} kù`,
      gameStart:    ()                 => 'Eré ti bẹ̀rẹ̀',
      youWin:       ()                 => 'O borí',
      youLose:      ()                 => 'O ṣẹ',
      draw:         ()                 => 'Àdéhùn',
    },

    ui: {
      toggle:       'Chess Ohun',
      language:     'Èdè',
      listening:    'Ń gbọ́...',
      notHeard:     'A kò gbọ́ ìgbésẹ̀ — jọ̀wọ́ gbìyànjú lẹ́ẹ̀kan sí i',
      illegal:      (move)            => `${move} kìí ṣe ìgbésẹ̀ tó tọ́`,
      confirm:      (move)            => `Ṣé o túmọ̀ sí ${move}? Sọ bẹ́ẹ̀ ni tàbí bẹ́ẹ̀ kọ`,
      voiceOn:      'Àwọn ìkéde ohun ti ṣí',
      voiceOff:     'Àwọn ìkéde ohun ti pa',
      micOn:        'Maikirofoonu ti ṣí — sọ ìgbésẹ̀ rẹ',
      micOff:       'Maikirofoonu ti pa',
      gridTitle:    'Yan ère',
      gridSquare:   'Yan square',
      scanning:     'Ìṣàyẹ̀wò yíyí ti ṣí',
    },

    voicePieces: {
      'ọmọ-ogun': 'p', 'ẹṣin': 'n', 'bíṣọ́ọ̀bù': 'b',
      'ilé-olódi': 'r', 'ayaba': 'q', 'ọba': 'k',
    },
    voiceFiles:   { 'alpha':'a','bravo':'b','charlie':'c','delta':'d','echo':'e','foxtrot':'f','golf':'g','hotel':'h' },
    voiceYes:     ['bẹ́ẹ̀ ni', 'sọtọ', 'tọ'],
    voiceNo:      ['bẹ́ẹ̀ kọ', 'rárá', 'fagilee'],
    voiceCastle:  ['ilé-olódi ọba'],
    voiceCastleQ: ['ilé-olódi ayaba'],
  },

  // ── Zulu ────────────────────────────────────────────────────────────────────
  'zu': {
    lang:         'isiZulu',
    speechCode:   'zu-ZA',
    altCode:      'en-ZA',

    pieces: {
      p: 'Izikhali',     // weapon/pawn
      n: 'Ihhashi',      // horse
      b: 'UBishop',      // bishop (loanword)
      r: 'Inqaba',       // fortress/rook
      q: 'INdlovukazi',  // queen/female elephant (great she-elephant)
      k: 'INkosi',       // chief/king
    },

    moves: {
      moveTo:       (piece, sq)        => `${piece} iya ku ${sq}`,
      captures:     (piece, sq)        => `${piece} uthatha ${sq}`,
      castle_k:     ()                 => 'Inqaba yoKumkani',
      castle_q:     ()                 => 'Inqaba yeNdlovukazi',
      promotes:     (piece, sq, promo) => `${piece} uba ${promo} ku ${sq}`,
      check:        ()                 => 'Isheki',
      checkmate:    (winner)           => `Isheki eliphelele — ${winner} unqobile`,
      stalemate:    ()                 => 'Umdlalo ulingene',
      draw:         ()                 => 'Umdlalo ulingene',
    },

    game: {
      yourTurn:     ()                 => 'Umjikelezo wakho',
      opponentTurn: (name)            => `Umjikelezo ka ${name}`,
      youPlay:      (color)           => `Udlala nge ${color}`,
      white:        'Mhlophe',
      black:        'Mnyama',
      timeLeft:     (t)               => `${t} kusele`,
      gameStart:    ()                 => 'Umdlalo uqalile',
      youWin:       ()                 => 'Unqobile',
      youLose:      ()                 => 'Ukalile',
      draw:         ()                 => 'Ulingene',
    },

    ui: {
      toggle:       'I-Chess Yezwi',
      language:     'Ulimi',
      listening:    'Ngizwa...',
      notHeard:     'Akuzwanga ukuhamba — zama futhi',
      illegal:      (move)            => `${move} akusilo ukuhamba okusemthethweni`,
      confirm:      (move)            => `Ingabe usho ${move}? Sho yebo noma cha`,
      voiceOn:      'Izaziso zezwi zivuliwe',
      voiceOff:     'Izaziso zezwi zivalwe',
      micOn:        'Umakrofoni uvuliwe — sho ukuhamba kwakho',
      micOff:       'Umakrofoni uvalwe',
      gridTitle:    'Khetha isicucu',
      gridSquare:   'Khetha ibhokisi',
      scanning:     'Ukuskana kweswitshi kuvuliwe',
    },

    voicePieces: {
      'izikhali': 'p', 'ihhashi': 'n', 'ubishop': 'b',
      'inqaba': 'r', 'indlovukazi': 'q', 'inkosi': 'k',
    },
    voiceFiles:   { 'alpha':'a','bravo':'b','charlie':'c','delta':'d','echo':'e','foxtrot':'f','golf':'g','hotel':'h' },
    voiceYes:     ['yebo', 'kulungile', 'cha'],
    voiceNo:      ['cha', 'iyah', 'khansela'],
    voiceCastle:  ['inqaba kumkani', 'inqaba emfishane'],
    voiceCastleQ: ['inqaba indlovukazi', 'inqaba ende'],
  },

  // ── Amharic ─────────────────────────────────────────────────────────────────
  'am': {
    lang:         'አማርኛ',
    speechCode:   'am-ET',
    altCode:      'en-ET',

    pieces: {
      p: 'ወታደር',        // soldier/pawn
      n: 'ፈረስ',          // horse/knight
      b: 'ዘውድ',          // bishop
      r: 'ቤተ መንግሥት',    // palace/rook
      q: 'ንግሥት',         // queen
      k: 'ንጉሥ',          // king
    },

    moves: {
      moveTo:       (piece, sq)        => `${piece} ወደ ${sq}`,
      captures:     (piece, sq)        => `${piece} ${sq}ን ይወስዳል`,
      castle_k:     ()                 => 'አጭር ካስሊንግ',
      castle_q:     ()                 => 'ረጅም ካስሊንግ',
      promotes:     (piece, sq, promo) => `${piece} ወደ ${promo} ይሸጋገራል ${sq}`,
      check:        ()                 => 'ቼክ',
      checkmate:    (winner)           => `ቼክሜት — ${winner} አሸነፈ`,
      stalemate:    ()                 => 'ዕኩልታ',
      draw:         ()                 => 'ዕኩልታ',
    },

    game: {
      yourTurn:     ()                 => 'የእርስዎ ተራ ነው',
      opponentTurn: (name)            => `የ${name} ተራ ነው`,
      youPlay:      (color)           => `${color} ሆነው ይጫወታሉ`,
      white:        'ነጭ',
      black:        'ጥቁር',
      timeLeft:     (t)               => `${t} ቀርቷል`,
      gameStart:    ()                 => 'ጨዋታ ተጀምሯል',
      youWin:       ()                 => 'አሸነፉ',
      youLose:      ()                 => 'ተሸነፉ',
      draw:         ()                 => 'ዕኩልታ',
    },

    ui: {
      toggle:       'የድምፅ ቼስ',
      language:     'ቋንቋ',
      listening:    'በማዳመጥ ላይ...',
      notHeard:     'እርምጃ አልተሰማም — እባክዎ እንደገና ይሞክሩ',
      illegal:      (move)            => `${move} ሕጋዊ እርምጃ አይደለም`,
      confirm:      (move)            => `${move} ማለትዎ ነው? አዎ ወይም አይ ይበሉ`,
      voiceOn:      'የድምፅ ማስታወቂያዎች ተከፍተዋል',
      voiceOff:     'የድምፅ ማስታወቂያዎች ተዘግተዋል',
      micOn:        'ማይክሮፎን ተከፍቷል — እርምጃዎን ይናገሩ',
      micOff:       'ማይክሮፎን ተዘግቷል',
      gridTitle:    'ቁርጥራጭ ይምረጡ',
      gridSquare:   'ካሬ ይምረጡ',
      scanning:     'የመቀያየሪያ ቅኝት ተሰቅሏል',
    },

    voicePieces: {
      'ወታደር': 'p', 'ፈረስ': 'n', 'ዘውድ': 'b',
      'ቤተ መንግሥት': 'r', 'ንግሥት': 'q', 'ንጉሥ': 'k',
    },
    voiceFiles:   { 'alpha':'a','bravo':'b','charlie':'c','delta':'d','echo':'e','foxtrot':'f','golf':'g','hotel':'h' },
    voiceYes:     ['አዎ', 'ትክክል', 'እሺ'],
    voiceNo:      ['አይ', 'አይደለም', 'ሰርዝ'],
    voiceCastle:  ['አጭር ካስሊንግ'],
    voiceCastleQ: ['ረጅም ካስሊንግ'],
  },
};

// Default language
const AAC_DEFAULT_LANG = 'en';

// Get strings for a language, falling back to English
function aacStrings(langCode) {
  return AAC_STRINGS[langCode] || AAC_STRINGS['en'];
}

// List of available languages for the picker
const AAC_LANGUAGES = Object.entries(AAC_STRINGS).map(([code, s]) => ({
  code,
  name: s.lang,
  speechCode: s.speechCode,
}));
