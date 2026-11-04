#!/bin/bash
# а1
set -uo pipefail
# а2
HERE="$(cd "$(dirname "$0")" && pwd)"
HOME_DIR="${TIER_DATA:-$HERE}"
NAME=${NAME:-run}
TITLE=${TITLE:-Обучение}
B=${B:-$HOME_DIR/tier}
M=${M:-$HOME_DIR/model/model.safetensors}
D=${D:-$HOME_DIR/work}
MODEL=${MODEL:-$(dirname "$M")}
PACK_SET=${PACK:-}; DATA=${1:-}
case "$DATA" in ''|/*) ;; *) DATA="$PWD/$DATA" ;; esac     # а3
mkdir -p "$D"; cd "$HOME_DIR"
export B M D MODEL
[ -x "$B" ] || { echo "Движок не собран: $B. Для запуска требуется сборка движка (команда: tierlm собрать)."; exit 1; }
[ -f "$M" ] || { echo "Нет весов: $M. Требуется указать папку модели в настройках."; exit 1; }

mats_tier(){ awk -v spec="$MATS" 'BEGIN{ split(spec,a,":"); nk=split(a[1],K,"+"); n=split(a[2],L,",")
    for(j=1;j<=nk;j++){ ks=(K[j]=="attn" ? "q_proj k_proj v_proj o_proj" : K[j]); nn=split(ks,KK," ")
      for(q=1;q<=nn;q++) for(i=1;i<=n;i++) want["layer_" L[i] "." KK[q]]=1 } }
  /^#/{ print; next } { if($1 in want){ print $1, 1, $3, $3; tot+=$3; k++ } else print $1, 2, $3, 0 }
  END{ printf "Целые матрицы: %s, слои %s (матриц: %d), обучается %d чисел, остальное заморожено.\n", a[1], a[2], k, tot > "/dev/stderr"; exit (k?0:1) }' "$1" > "$1.t" && mv "$1.t" "$1"; }   # б1
say(){ printf '\n\033[1;96m%s\033[0m\n' "$*"; }
stage(){ printf '%s\n' "$1" > "$D/${NAME}_stage.txt"; rm -f "$D/${NAME}_stagep.txt"; }   # г1
num(){ awk -v k="$1" '$1==k{print $2}' "$D/${NAME}_stats.txt" 2>/dev/null; }
# е1
notify_end(){ local rc=$? st b msg
  [ "${TIER_NOTIFY:-0}" = 1 ] && command -v notify-send >/dev/null 2>&1 || return 0
  { [ "$rc" = 143 ] || [ "$rc" = 130 ]; } && return 0
  if [ -n "${TIER_UI_PID:-}" ] && [ -f "$TIER_UI_PID" ] && kill -0 "$(cat "$TIER_UI_PID" 2>/dev/null)" 2>/dev/null; then return 0; fi
  st=$(cat "$D/${NAME}_stage.txt" 2>/dev/null)
  if [ "$rc" = 0 ] && [ "$st" = Завершено ]; then
    b=$(awk -F'\t' '$2=="Оценка"&&$14!=""&&(m==""||$14<m){m=$14} END{print m}' "$D/${NAME}_log.tsv" 2>/dev/null)
    if [ "${TIER_LANG:-ru}" = en ]; then msg="Training “$NAME” finished.${b:+ Best loss: $b.}"; else msg="Обучение «$NAME» завершено.${b:+ Лучшая потеря (loss): $b.}"; fi
    notify-send -a TierLM -i tierlm TierLM "$msg"
  else
    if [ "${TIER_LANG:-ru}" = en ]; then msg="Training “$NAME” stopped with an error (code $rc)."; else msg="Обучение «$NAME» остановлено из-за ошибки (код $rc)."; fi
    notify-send -a TierLM -i tierlm -u critical TierLM "$msg"
  fi; return 0; }
trap notify_end EXIT

# ж1
if [ -n "$DATA" ] || [ ! -f "$D/${NAME}_tokens.bin" ]; then
    [ -n "$DATA" ] || { echo "Нет токенов, и датасет не передан. Формат запуска: bash auto.sh датасет.jsonl"; exit 1; }
    stage Токены; say "Подготовка токенов из $DATA"
    NAME=$NAME WORK=$D MODEL=$MODEL "${TIER_PY:-python3}" "$HERE/prepare.py" "$DATA" || exit 1
fi
EX=$(num examples); MED=$(num median); P99=$(num p99); LOSSTOK=$(num loss_tokens)
[ -n "${EX:-}" ] || { echo "Нет сводки $D/${NAME}_stats.txt. Требуется повторная подготовка токенов."; exit 1; }

# ж2
PACK=$("${TIER_PY:-python3}" "$HERE/hwprobe.py" --pack "$P99" "$EX" "${TIER_MODEL_DESC:-$MODEL}" 2>/dev/null)
case "$PACK" in ''|*[!0-9]*)   # ж3
  if   [ "$EX" -le 256 ]; then PACK=0
  elif [ "$P99" -le 160 ]; then PACK=512
  elif [ "$P99" -le 400 ]; then VR=$(awk '$1=="vram_mib"{print $2}' "${TIER_CONF:-$HOME/.config/tier.conf}" 2>/dev/null); [ "${VR:-0}" -ge 12000 ] && PACK=1024 || PACK=512   # ж4
  else PACK=0; fi ;;
esac
# ж5
if   [ "$EX" -le 64 ];   then EPOCHS=${EPOCHS:-12}
elif [ "$EX" -le 512 ];  then EPOCHS=${EPOCHS:-6}
elif [ "$EX" -le 4096 ]; then EPOCHS=${EPOCHS:-3}
else EPOCHS=${EPOCHS:-2}; fi
PACK=${PACK_SET:-$PACK}
TOK=$D/${NAME}_tokens.bin
PACK_U=$PACK   # ж6
# з1
make_retain(){
  [ -s "$D/${NAME}_retain.bin" ] && return
  REPLAY_DIR="$HOME_DIR/replay" B=$B M=$M MODEL=$MODEL X="${X:-}" "${TIER_PY:-python3}" "$HERE/replay.py" retain "$D/${NAME}_retain.bin" 64 >/dev/null 2>>"$D/${NAME}_conflicts.log"
}
# и1
find_conflicts(){
  [ -n "${CONFLICTS_DONE:-}" ] && return; CONFLICTS_DONE=1; rm -f "$D/${NAME}_conflicts.txt"
  [ "${CONFLICT:-1}" != 0 ] || return
  NC=$(REPLAY_DIR="$HOME_DIR/replay" B=$B M=$M MODEL=$MODEL X="${X:-}" "${TIER_PY:-python3}" "$HERE/replay.py" topics "$D/${NAME}_tokens.bin" "$D/${NAME}_conflicts.txt" 2>>"$D/${NAME}_conflicts.log" || echo 0)
  say "Противоречий с данными в обычных ответах: ${NC:-0} (вопросы на темы данных), в подмешивание и в замер порчи они не идут."
  [ -s "$D/${NAME}_conflicts.txt" ] && export REPLAY_EXCLUDE="$D/${NAME}_conflicts.txt"
}
REPLAY=${REPLAY:-1}; rm -f "$D/${NAME}_mix.bin" "$D/${NAME}_replay.txt"   # и2
if [ "$REPLAY" != 0 ] && [ -f "$HERE/replay.jsonl" ]; then
  # и3
  find_conflicts
  [ -s "$D/${NAME}_conflicts.txt" ] && export REPLAY_EXCLUDE="$D/${NAME}_conflicts.txt"
  make_retain
  [ -s "$D/${NAME}_retain.bin.txt" ] && export REPLAY_HOLDOUT="$D/${NAME}_retain.bin.txt"
  stage Подмешивание; say "Подмешивание: к каждому примеру добавляются обычные ответы исходной модели, ×$REPLAY к токенам датасета."
  OUTR=$(STAGE_PROG="$D/${NAME}_stagep.txt" B=$B M=$M MODEL=$MODEL X="${X:-}" REPLAY_DIR="$HOME_DIR/replay" REPLAY_SCENES="$D/${NAME}_scenes.json" "${TIER_PY:-python3}" "$HERE/replay.py" mix \
           "$D/${NAME}_tokens.bin" "$D/${NAME}_mix.bin" "$REPLAY" "$PACK") || { echo "Подмешивание не выполнено, обучение продолжается без него."; OUTR=; }
  printf '%s\n' "$OUTR" | grep '^Подмешивание' || true
  if [ -n "$OUTR" ]; then TOK=$D/${NAME}_mix.bin; PACK=$(printf '%s\n' "$OUTR" | awk '$1=="pack"{print $2}'); echo "$REPLAY" > "$D/${NAME}_replay.txt"; fi
fi
export TOK
# и4
if [ -n "${MATS:-}" ] && [ "${MATS_AUTO:-0}" = 1 ]; then
  stage Замер; say "Замер: определение слоёв, на которые данные влияют сильнее всего"
  KIND=${MATS%%:*}; NL=$(awk '$1=="layers"{print $2}' "${TIER_MODEL_DESC:-/dev/null}" 2>/dev/null)
  [ -n "$NL" ] || NL=$("${TIER_PY:-python3}" "$HERE/arch.py" desc "$MODEL" | awk '$1=="layers"{print $2}')
  rm -f "$D/probe_"*.bin "$D/probe_tier.txt"
  "$B" "$M" /dev/null 8 "$D/probe_adam.bin" "$D/probe_w.bin" 1e-12 9999 1 --retier ${X:-} --tier-file "$D/probe_tier.txt" >/dev/null 2>&1
  "$B" "$M" /dev/null 8 "$D/probe_adam.bin" "$D/probe_w.bin" 1e-12 9999 1 ${X:-} --tier-file "$D/probe_tier.txt" --data "$D/${NAME}_tokens.bin" \
      --steps 6 --pack "$PACK_U" --shuffle --flush-every 0 --grad-probe 2>/dev/null | grep '^gprobe' > "$D/${NAME}_gprobe.txt"
  rm -f "$D/probe_"*.bin "$D/probe_tier.txt"
  LAY=$("${TIER_PY:-python3}" "$HERE/replay.py" layers "$KIND" "${NL:-0}" < "$D/${NAME}_gprobe.txt" 2>"$D/${NAME}_gprobe_why.txt")
  if [ -n "$LAY" ]; then say "Слои правки по результатам замера: $LAY (по долям глубины было: ${MATS#*:})."; MATS="$KIND:$LAY"
  else say "Замер не выполнен, слои определены по долям глубины: ${MATS#*:}."; fi
fi
# и5
rm -f "$D/${NAME}_mode.txt"
if [ -n "${MATS:-}" ] && [ "${NSGD:-0}" = 1 ]; then
  X="${X:-} --nsgd"; [ "${POSDW:-0}" = 1 ] && X="$X --pos-dw"
  if [ "${WHITEN:-1}" != 0 ]; then
    stage Защита; say "Защита общих знаний: исходная модель читает обычный текст, выполняется поиск главных направлений входа $MATS."
    WH=$(STAGE_PROG="$D/${NAME}_stagep.txt" B=$B M=$M MODEL=$MODEL X="$X" REPLAY_DIR="$HOME_DIR/replay" "${TIER_PY:-python3}" "$HERE/replay.py" whiten "$MATS" "${WHITEN_A:-0.3}" "${WHITEN_FRAC:-0.1}") \
      || { echo "Защита общих знаний не выполнена, обучение продолжается без неё."; WH=; }
    [ -n "$WH" ] && X="$X --whiten $WH"
  fi
  printf 'MATS=%s\nNSGD=1\nPOSDW=%s\nWHITEN=%s\nWHITEN_FRAC=%s\n' "$MATS" "${POSDW:-0}" "${WHITEN:-1}" "${WHITEN_FRAC:-0.1}" > "$D/${NAME}_mode.txt"
fi
# и6
if [ -n "${LORA:-}" ]; then
  X="${X:-} --lora $LORA"; printf 'LORA=%s\n' "$LORA" > "$D/${NAME}_mode.txt"
  say "LoRA $LORA: исходные веса заморожены, учатся только адаптеры."
fi
# и7
CONF=${TIER_CONF:-$HOME/.config/tier.conf}; MID=$(basename "$MODEL")
if [ -z "${LORA:-}" ] && [ "${CALIB:-1}" != 0 ] && ! grep -q "^model $MID " "$CONF" 2>/dev/null; then
  stage Замер; say "Замер видеопамяти для модели $MID"
  VR=$(awk '$1=="vram_mib"{print $2}' "$CONF" 2>/dev/null)
  if [ -n "$VR" ]; then
    CC="$D/calib.conf"; awk -v t=$(( VR * 1048576 / 12 / 10 )) '$1=="tier1_params"{print $1, t; next} {print}' "$CONF" > "$CC"
    rm -f "$D/probe_"*.bin "$D/probe_tier.txt"
    TIER_CONF="$CC" "$B" "$M" /dev/null 8 "$D/probe_adam.bin" "$D/probe_w.bin" 1e-12 9999 1 --retier ${X:-} --pack "$PACK_U" --tier-file "$D/probe_tier.txt" >/dev/null 2>&1
    TIER_CONF="$CC" "$B" "$M" /dev/null 8 "$D/probe_adam.bin" "$D/probe_w.bin" 1e-12 9999 1 ${X:-} --tier-file "$D/probe_tier.txt" --data "$D/${NAME}_tokens.bin" \
        --steps 4 --pack "$PACK_U" --shuffle --flush-every 0 --calibrate-conf "$CONF" "$MID" >/dev/null 2>&1 && say "Бюджет обучаемой части снят замером." || say "Замер видеопамяти не выполнен, бюджет по оценке."
    rm -f "$D/probe_"*.bin "$D/probe_tier.txt" "$CC"
  fi
fi
# и8
if [ -z "${MATS:-}" ] && [ -z "${LORA:-}" ] && [ "${ROWSEL:-1}" != 0 ]; then
  stage Замер; say "Замер: выбор строк среза по градиенту данных"
  rm -f "$D/probe_"*.bin "$D/probe_tier.txt"
  "$B" "$M" /dev/null 8 "$D/probe_adam.bin" "$D/probe_w.bin" 1e-12 9999 1 --retier ${X:-} --tier-file "$D/probe_tier.txt" >/dev/null 2>&1
  "$B" "$M" /dev/null 8 "$D/probe_adam.bin" "$D/probe_w.bin" 1e-12 9999 1 ${X:-} --tier-file "$D/probe_tier.txt" --data "$D/${NAME}_tokens.bin" \
      --steps ${ROWSEL_STEPS:-6} --pack "$PACK_U" --shuffle --flush-every 0 --row-probe "$D/${NAME}_rows.txt" >/dev/null 2>&1
  rm -f "$D/probe_"*.bin "$D/probe_tier.txt"
  if [ -s "$D/${NAME}_rows.txt" ]; then X="${X:-} --row-scores $D/${NAME}_rows.txt"; [ "${ROWSEL:-1}" = 2 ] && X="$X --row-global"; say "Строки среза выбраны по замеру."
  else say "Замер строк не выполнен, срез берёт первые строки."; fi
fi
export X
# и9
if [ "$TOK" != "$D/${NAME}_tokens.bin" ]; then PER=$(printf '%s\n' "$OUTR" | awk '$1=="steps"{print $2}')
elif [ "$PACK" = 0 ]; then PER=$EX; else PER=$(( (EX*MED + PACK - 1) / PACK )); fi
[ "$PER" -lt 1 ] && PER=1
TOTAL=$(( PER * EPOCHS ))
say "Датасет: примеров $EX, медиана $MED токенов, p99 $P99, токенов в расчёте потери (loss) $LOSSTOK. Упаковка $PACK, эпох $EPOCHS, шагов около $TOTAL."

# к1
LRF=$D/${NAME}_lrfind.txt
if [ -z "${LR:-}" ]; then
  stage Развёртка; say "Подбор LR: 40 коротких шагов"
  for TRY in 1 2 3; do   # к2
  rm -f "$D/probe_"*.bin "$D/probe_tier.txt" "$LRF"
  # к3
  "$B" "$M" /dev/null 8 "$D/probe_adam.bin" "$D/probe_w.bin" 1e-4 9999 1 \
      --retier ${X:-} --tier-file "$D/probe_tier.txt" >/dev/null 2>&1
  [ -z "${MATS:-}" ] || mats_tier "$D/probe_tier.txt" 2>/dev/null || { echo "Матрицы не найдены: $MATS."; exit 1; }   # к4
  # к5
  LB=; command -v stdbuf >/dev/null && LB="stdbuf -oL"
  "$B" "$M" /dev/null 8 "$D/probe_adam.bin" "$D/probe_w.bin" 1e-4 9999 1 ${X:-} \
      --tier-file "$D/probe_tier.txt" --data "$D/${NAME}_tokens.bin" \
      --steps 40 --lr-find 40 --pack "$PACK_U" --shuffle --flush-every 0 2>&1 \
    | $LB tr '\r' '\n' | awk -v f="$LRF" -v lf="$D/${NAME}_load.txt" '/^lrfind/{ print > f; fflush(f); if(!z){ printf "" > lf; close(lf); z=1 } next }
        /Загрузка [0-9]/{ x=$0; sub(/^.*Загрузка /,"",x); print x > lf; close(lf) }'   # к6
  RCL=${PIPESTATUS[0]}
  [ "$RCL" = 76 ] && { say "Видеопамяти не хватило: запас увеличен, подбор LR повторяется с новой раскладкой."; continue; }
  break; done
  rm -f "$D/probe_"*.bin "$D/probe_tier.txt"
  # к7
  FB=2e-5; [ -n "${LORA:-}" ] && FB=2e-4
  LR=$(awk -v fb="$FB" '{lr[NR]=$3; l[NR]=$4} END{
          if(NR<8){ print fb; exit }
          for(i=2;i<NR;i++) sm[i]=(l[i-1]+l[i]+l[i+1])/3            # сглаживание тройкой: одиночный выброс не решает
          mi=0; mv=1e30
          for(i=2;i<NR;i++) if(sm[i]<mv){ mv=sm[i]; mi=i }
          if(!mi || sm[2]-mv < 0.05){ print fb; exit }               # потеря так и не падала — развёртка ничего не сказала
          printf "%.3g", lr[mi]/10 }' "$LRF")
  [ -n "$LR" ] || LR=$FB
  # к8
  if [ "${LR_PROBE:-1}" != 0 ] && [ -s "$D/${NAME}_eval.bin" ]; then
    stage Пробы; say "Пробы шага обучения: выучено против испорчено"
    LMAX=$(awk '{lr[NR]=$3; l[NR]=$4} END{ if(NR<8){ exit } for(i=2;i<NR;i++){ sm[i]=(l[i-1]+l[i]+l[i+1])/3; if(!mi||sm[i]<m){ m=sm[i]; mi=i } }
                for(i=mi+1;i<NR;i++) if(sm[i]>3*m && lr[i]>0){ print lr[i]; exit } print lr[NR-1] }' "$LRF")   # к9
    [ "${CONFLICT:-1}" != 0 ] && [ -f "$HERE/replay.jsonl" ] && find_conflicts
    RET="$D/${NAME}_retain.bin"; make_retain
    evl(){ "$B" "$M" /dev/null 8 /dev/null "$1" 1e-5 9999 1 ${X:-} --tier-file "$D/probe_tier.txt" --data "$2" --eval --steps "$3" 2>/dev/null | grep -a '=== EVAL' | sed 's/.*mean loss \([0-9.]*\).*/\1/'; }
    NEV=$(od -An -t d4 -N4 "$D/${NAME}_eval.bin" | tr -d " "); NRET=$(od -An -t d4 -N4 "$RET" 2>/dev/null | tr -d " ")
    HS=$(( PER / 2 )); [ "$HS" -lt 8 ] && HS=8
    if [ -n "$LMAX" ]; then
      rm -f "$D/probe_"*.bin "$D/probe_tier.txt"
      "$B" "$M" /dev/null 8 "$D/probe_adam.bin" "$D/probe_w.bin" 1e-5 9999 1 --retier ${X:-} --tier-file "$D/probe_tier.txt" >/dev/null 2>&1
      [ -z "${MATS:-}" ] || mats_tier "$D/probe_tier.txt" 2>/dev/null
      # л1
      R0=; BEST=; BS=; FIRST=; RA=(); [ -s "$RET" ] && { RA=(--eval-data "$RET"); FIRST=--eval-first; }
      el(){ printf '%s\n' "$OUTP" | grep -a "^=== EVAL step $1" | head -1 | sed 's/.*mean loss \([0-9.]*\).*/\1/'; }
      declare -A SCD=(); ORDER="100 30 300"; DONE_=" "; BD=
      while [ -n "$ORDER" ]; do DIV=${ORDER%% *}; ORDER=${ORDER#"$DIV"}; ORDER=${ORDER# }; DONE_="$DONE_$DIV "
        C=$(awk -v a="$LMAX" -v d=$DIV 'BEGIN{printf "%.3g", a/d}')
        rm -f "$D/probe_adam.bin" "$D/probe_w.bin"
        for PT_ in 1 2; do   # м1
          rm -f "$D/probe_adam.bin" "$D/probe_w.bin"
          OUTP=$("$B" "$M" /dev/null 8 "$D/probe_adam.bin" "$D/probe_w.bin" "$C" 9999 1 ${X:-} --tier-file "$D/probe_tier.txt" --data "$TOK" \
              --steps $HS --pack "$PACK" --shuffle --flush-every 0 --eval-data "$D/${NAME}_eval.bin" "${RA[@]}" --eval-every $HS $FIRST 2>/dev/null | grep -a '^=== EVAL step'; echo "rc ${PIPESTATUS[0]}")
          printf '%s\n' "$OUTP" | grep -q '^rc 76$' || break
          say "Проба шага $C: не хватило видеопамяти, раскладка перестроена, повтор."
          "$B" "$M" /dev/null 8 "$D/probe_adam.bin" "$D/probe_w.bin" 1e-5 9999 1 --retier ${X:-} --tier-file "$D/probe_tier.txt" >/dev/null 2>&1
        done
        [ -n "$FIRST" ] && R0=$(el "0 set 1"); FIRST=
        LU=$(el "$HS:"); LR_=$( [ ${#RA[@]} -gt 0 ] && el "$HS set 1" )
        if [ -n "$OUTP" ] && [ ${#RA[@]} -gt 0 ] && [ -z "$LR_" ] && ! printf '%s\n' "$OUTP" | grep -q ' set 1:'; then
          # м2
          LU=$(evl "$D/probe_w.bin" "$D/${NAME}_eval.bin" "${NEV:-1}"); LR_=$(evl "$D/probe_w.bin" "$RET" "${NRET:-1}")
          [ -n "$R0" ] || R0=$(evl "$D/probe_none.bin" "$RET" "${NRET:-1}")
        fi
        # м3
        [ ${#RA[@]} -gt 0 ] && { [ -z "$LR_" ] || [ -z "$R0" ]; } && LU=
        SC=$(awk -v u="${LU:-99}" -v r="${LR_:-0}" -v r0="${R0:-0}" 'BEGIN{ d=r-r0; if(d<0) d=0; printf "%.4f", u+d }')
        say "Проба шага $C: отложенные ${LU:-?}, обычные ${LR_:-?} (исходная ${R0:-?}), счёт $SC."
        if [ -n "$LU" ] && { [ -z "$BS" ] || awk -v a="$SC" -v b="$BS" 'BEGIN{exit !(a<b)}'; }; then BS=$SC; BEST=$C; BD=$DIV; fi
        [ -n "$LU" ] && SCD[$DIV]=$SC
        if [ -z "$ORDER" ]; then   # м4
          case "${BD:-}" in 30) [[ "$DONE_" == *" 10 "* ]] || ORDER=10 ;; 300) [[ "$DONE_" == *" 1000 "* ]] || ORDER=1000 ;; esac
          [ -n "${BD:-}" ] || { [[ "$DONE_" == *" 10 "* ]] || ORDER="10 1000"; }   # м5
        fi
      done
      if [ -z "${LORA:-}" ] && [ -z "${MATS:-}" ] && [ "${WIDTH_PROBE:-1}" != 0 ] && [ -n "$BEST" ] && [ -n "$R0" ]; then
        WF=1; WB=$BEST; FB=$BS; NB=; NBS=; NW=   # м6
        for F in 0.25 0.0625; do
          rm -f "$D/probe_tier.txt"
          "$B" "$M" /dev/null 8 "$D/probe_adam.bin" "$D/probe_w.bin" 1e-5 9999 1 --retier ${X:-} --slice-frac $F --tier-file "$D/probe_tier.txt" >/dev/null 2>&1 || continue
          for K in 1 3; do
            C=$(awk -v a="$WB" -v k=$K 'BEGIN{printf "%.3g", a*k}')
            rm -f "$D/probe_adam.bin" "$D/probe_w.bin"
            OUTP=$("$B" "$M" /dev/null 8 "$D/probe_adam.bin" "$D/probe_w.bin" "$C" 9999 1 ${X:-} --tier-file "$D/probe_tier.txt" --data "$TOK" \
                --steps $HS --pack "$PACK" --shuffle --flush-every 0 --eval-data "$D/${NAME}_eval.bin" "${RA[@]}" --eval-every $HS 2>/dev/null | grep -a '^=== EVAL step')
            LU=$(el "$HS:"); LR_=$( [ ${#RA[@]} -gt 0 ] && el "$HS set 1" )
            [ ${#RA[@]} -gt 0 ] && [ -z "$LR_" ] && LU=
            SC=$(awk -v u="${LU:-99}" -v r="${LR_:-0}" -v r0="${R0:-0}" 'BEGIN{ d=r-r0; if(d<0) d=0; printf "%.4f", u+d }')
            say "Проба ширины $F, шаг $C: отложенные ${LU:-?}, обычные ${LR_:-?}, счёт $SC."
            if [ -n "$LU" ] && awk -v a="$SC" -v b="$BS" 'BEGIN{exit !(a<b)}'; then BS=$SC; BEST=$C; WF=$F; fi
            # м7
            if [ -n "$LU" ] && awk -v a="$SC" -v b="$FB" 'BEGIN{exit !(a<=b*1.02)}' && { [ "$NW" != "$F" ] || awk -v a="$SC" -v b="$NBS" 'BEGIN{exit !(a<b)}'; }; then NW=$F; NB=$C; NBS=$SC; fi
          done
        done
        [ -n "$NW" ] && { WF=$NW; BEST=$NB; BS=$NBS; }
        if [ "$WF" != 1 ]; then X="${X:-} --slice-frac $WF"; export X; say "Ширина среза по пробам: $WF бюджета."; else say "Ширина среза по пробам: весь бюджет."; fi
      fi
      rm -f "$D/probe_"*.bin "$D/probe_tier.txt"
      [ -n "$BEST" ] && LR=$BEST
    fi
  fi
  say "Выбран LR $LR (результаты подбора: $LRF)."
else
  say "LR задан вручную: $LR."
fi

# н1
[ -n "${EV:-}" ] || { EV=$(( PER / 2 )); [ "$EV" -lt 1 ] && EV=1; }
[ -n "${STOP_AFTER:-}" ] || { HE=$(( (EPOCHS + 1) / 2 )); [ "$HE" -lt 1 ] && HE=1; STOP_AFTER=$(( (PER + EV - 1) / EV * HE )); [ "$STOP_AFTER" -lt 2 ] && STOP_AFTER=2; }
[ -n "${FL:-}" ] || FL=$(( EV + 1 ))     # н2
say "Валидация каждые $EV шагов, снимок весов каждые $FL шагов, остановка после $STOP_AFTER проверок без улучшения (половина эпох)."

# о1
say "Обучение: эпох $EPOCHS, LR $LR, упаковка $PACK."
for TRY in 1 2; do   # о2
NAME=$NAME TITLE=$TITLE M=$M D=$D TOK=$TOK MATS=${MATS:-} EPOCHS=$EPOCHS PACK=$PACK LR=$LR EV=$EV FL=$FL \
  B=$B STOP_AFTER=$STOP_AFTER bash "$HERE/train.sh"; RCT=$?
[ "$RCT" = 76 ] || break
say "Видеопамяти не хватило: запас увеличен, обучение начинается заново с новой раскладкой."
rm -f "$D/tier_assignment_${NAME}_act.txt" "$D/adam_state_${NAME}.bin" "$D/${NAME}_tier1.bin" "$D/${NAME}_step.txt"
done
# о3
if [ "$RCT" = 0 ] && [ "${SELFCHECK:-1}" != 0 ]; then
  stage Самопроверка; say "Самопроверка: сравнение с исходной моделью"
  CK="$D/${NAME}_tier1_best.bin"; [ -s "$CK" ] || CK="$D/${NAME}_tier1.bin"; TF="$D/tier_assignment_${NAME}_act.txt"
  ev(){ "$B" "$M" /dev/null 8 /dev/null "$1" 1e-5 9999 1 ${X:-} --tier-file "$TF" --data "$2" --eval --steps "$(od -An -t d4 -N4 "$2" | tr -d " ")" 2>/dev/null | grep -a '=== EVAL' | sed 's/.*mean loss \([0-9.]*\).*acc \([0-9.]*\).*/\1 \2/'; }
  make_retain
  : > "$D/${NAME}_selfcheck.txt"
  for set in "отложенные:$D/${NAME}_eval.bin" "обычные:$D/${NAME}_retain.bin"; do nm=${set%%:*}; f=${set#*:}; [ -s "$f" ] || continue
    b=$(ev "$D/${NAME}_none.bin" "$f"); t=$(ev "$CK" "$f"); [ -n "$b" ] && [ -n "$t" ] || continue
    line=$(awk -v n="$nm" -v b="$b" -v t="$t" 'BEGIN{ split(b,x," "); split(t,y," "); printf "%s: потеря исходной %.4f, обученной %.4f (%+.0f %%), точность %.3f -> %.3f", n, x[1], y[1], 100*(y[1]-x[1])/(x[1]>0?x[1]:1), x[2], y[2] }')
    echo "$line" >> "$D/${NAME}_selfcheck.txt"; say "Самопроверка, $line."; done
fi
exit $RCT
