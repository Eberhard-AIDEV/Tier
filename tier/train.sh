#!/bin/bash
# б1
NAME=${NAME:-run}                    # б2
TITLE=${TITLE:-Обучение}                  # б3
DATA=${DATA:-./dataset.jsonl}  # б4
M=${M:-./model/model.safetensors}   # б5
D=${D:-./work}                   # б6
TOK=${TOK:-}                          # б7
[ -n "$TOK" ] || { TOK=$D/${NAME}_mix.bin; [ -f "$TOK" ] || TOK=$D/${NAME}_tokens.bin; }   # б8
B=${B:-./tier}                        # б9
LR=${LR:-1e-5}                        # б10
X=${X:-}                              # б11
PACK=${PACK:-}                        # б12
[ -n "$PACK" ] || [ ! -s "$D/${NAME}_stats.txt" ] || PACK=$(st(){ awk -v k="$1" '$1==k{print $2}' "$D/${NAME}_stats.txt" 2>/dev/null; }
    python3 "$(dirname "$0")/hwprobe.py" --pack "$(st p99)" "$(st examples)" "${TIER_MODEL_DESC:-$(dirname "$M")}" 2>/dev/null)
case "$PACK" in ''|*[!0-9]*) PACK=512 ;; esac
EPOCHS=${EPOCHS:-2}                   # б13
EV=${EV:-100}                         # б14
FL=${FL:-50}                          # б15
STOP_AFTER=${STOP_AFTER:-0}           # б16
EVERY=${EVERY:-50}                    # б17
# в1
set -eo pipefail; T0=$(date +%s)   # в2
mkdir -p "$D"   # в3
trap 'printf "\033[?25h" >&2' EXIT   # в4
TIER=$D/tier_assignment_${NAME}_act.txt; ADAM=$D/adam_state_${NAME}.bin; CKPT=$D/${NAME}_tier1.bin
LT=$D/${NAME}_log.tsv   # в5
[ -f $LT ] || printf 'Время\tСобытие\tШаг\tЭпоха\tПервый_пример\tПримеров\tТокенов\tТокенов_в_loss\tLoss\tPPL\tAcc_%%\tLoss_ср50\tAcc_ср50_%%\tLoss_отложенные\tЛучший_отложенный\t|g|\t|g|_макс\tLR\tСек_шаг\tПрямой_с\tГоловка_с\tОбратный_Adam_с\tТок_в_с\tПрошло\tОсталось\tVRAM_пик_МиБ\tVRAM_всего_МиБ\tЖивых_МиБ\tБлоков\tОбласть_МиБ\tМимо_области\tT_край_°C\tT_горячая_°C\tT_память_°C\tВент_об_мин\tОЗУ_кэш_ГиБ\tОбучаемая_часть_в_VRAM\tСообщение\n' > $LT
lg(){ printf '%s\t%s' "$(date '+%F %T')" "$1" >> $LT; printf '\t%.0s' {1..35} >> $LT; printf '\t%s\n' "$2" >> $LT; }   # г1
stage(){ printf '%s\n' "$1" > "$D/${NAME}_stage.txt"; }   # д1
npk(){ python3 - "$1" "$PACK" <<'PY'
import struct,sys
f=open(sys.argv[1],'rb'); n,=struct.unpack('<i',f.read(4)); B=int(sys.argv[2]); packs=0; ct=0; cur=0
for _ in range(n):
    L,P=struct.unpack('<ii',f.read(8)); f.seek(4*L*(2 if P<0 else 1),1)   # P<0: следом ещё метки повтора
    if B<=0: packs+=1; continue
    if cur and ct+L>B: packs+=1; ct=0; cur=0
    ct+=L; cur+=1
print(packs+(1 if cur else 0))
PY
}   # е1
N=$(npk $TOK); NE=$(npk $D/${NAME}_eval.bin)
BASE=$(cat $D/${NAME}_step_base.txt 2>/dev/null || echo 0)   # е2
TOTAL=$((BASE+N*EPOCHS))
DASH=$(cat <<'AWK'
function bar(p,c,  w,x,f,r,b,i,part){ w=bw; if(p>1)p=1; if(p<0)p=0     # сплошные блоки без зазоров; край — восьмушками, чтобы двигался плавно
  part[1]="▏"; part[2]="▎"; part[3]="▍"; part[4]="▌"; part[5]="▋"; part[6]="▊"; part[7]="▉"
  x=p*w; f=int(x); r=int((x-f)*8)
  b=""; for(i=0;i<f;i++) b=b "█"
  if(f<w && r>0){ b=b part[r]; f++ }
  b=c b "\033[0m"; for(i=f;i<w;i++) b=b " "; return b }   # остаток — пустое место: без серой подложки чище
function hm(x){ return x<60?sprintf("%d с",x):(x<3600?sprintf("%d мин %02d с",x/60,x%60):sprintf("%d ч %02d мин",x/3600,(x%3600)/60)) }
function vis(t){ gsub(/\033\[[0-9;]*m/,"",t); return length(t) }
function pad(t,w){ while(vis(t)<w) t=t " "; return t }
function B(x){ return "\033[1;97m" x "\033[0m" }
function D(x){ return "\033[0;38;5;110m" x "\033[0m" }   # подписи мягким голубым: читаются и на прозрачном фоне
function put(key,r,c,w,t){ t=pad(t,w); if(prev[key]==t) return; prev[key]=t          # только изменившееся значение:
  F=F "\033[" (H-r) "A\033[" (c+1) "G" t "\r\033[" (H-r) "B" }                       # вверх в свою строку, в свой столбец, обратно под панель
function frame(  i,lab){ vw=(cols>=100?14:11); cw=11+vw; nc=int((cols-3)/cw); if(nc>4)nc=4; if(nc<1)nc=1   # ни одна строка не длиннее окна:
  bw=cols-8-2-28-3; if(bw>56)bw=56; if(bw<8)bw=8; H=4+int((17+nc-1)/nc)                               # иначе перенос сдвигает клетки
  printf "\033[?25l" > "/dev/stderr"
  split("Потеря,Средняя,Перплексия,Точность,|g|,|g| макс,LR,Минимум,Время,Осталось,Шаг,Ток/с,VRAM,Температура,Вентилятор,Валидация,Диск",lab,",")
  F="\n  \033[1;96m" ttl "\033[0m\n  " D("Шаг") "\n\n"
  for(i=0;i<17;i++){ if(i%nc==0) F=F "  "; F=F pad(D(lab[i+1]),11) sprintf("%" vw "s",""); if(i%nc==nc-1||i==16) F=F "\n" }
  printf "%s", F > "/dev/stderr"; drawn=1 }
function cellv(i,x){ put("c" i, 4+int(i/nc), 2+(i%nc)*cw+11, vw, B(x)) }
function disk(  f,l,z,t){ if(dev=="") return "—"; t=tnow?tnow:systime(); if(dt0!="" && t-dt0<0.5) return dtxt   # файл статистики читаем раз в полсекунды, а не каждый кадр
  f="/sys/block/" dev "/stat"; if((getline l < f)>0){ split(l,z," "); close(f)
    if(dt0!="" && t-dt0>=0.5){ drate=(z[3]-ds0)*512/(t-dt0)/1048576; ds0=z[3]; dt0=t } else if(dt0==""){ ds0=dsb=z[3]; dt0=t }; dgb=(z[3]-dsb)*512/1073741824 }
  dtxt=(drate<1 ? "Простой" : sprintf("%.0f МБ/с",drate)) sprintf(" · %.0f ГБ",dgb); return dtxt }  # прочитано с диска с начала запуска: загрузка ОЗУ-кэша — ~19,4 ГиБ
function draw(  now,el,eta,ph,pp,k,i,a,aa,sp){
  if(!drawn) frame()
  now=systime(); el=st0+now-t0; step=s+n; sp=(spe>0?spe:(m>0?Tm/m:0)); eta=(m>0?(tot-step+1)*Tm/m:-1)   # полоса — по скользящей, остаток — по общему среднему
  # Ход шага ведём по времени, а не по событиям движка: событий на шаг всего четыре, и полоса от них прыгала ступенями.
  # Длительность шага известна, панель рисует 60 раз в секунду — заполнение выходит ровным. Назад не отматываем, обнуляем на новом шаге.
  # Часов двое: отметка движка (tick) и свои (systime). Смешивать их в одной переменной нельзя — начало шага
  # запишется по одним, кадры придут по другим, и разность навсегда уйдёт в минус, а полоса застынет на нуле.
  # Отметка из тиков хороша долями секунды, но если поток тиков прервался, она застывает — и разность «сейчас минус
  # начало шага» навсегда обращается в ноль. Поэтому считаем её годной, только пока она меняется.
  if(tnow!=ptn){ ptn=tnow; tseen=now }
  fresh=(tnow && tseen && now-tseen<2)          # тик считается свежим две секунды; дальше идём по своим часам
  tfr=(fresh?tnow:now); dom=(fresh?1:0)
  # на смене шага гасим и счётчики этапов: иначе хвост прошлого шага (40/40/40/40) читается как «всё готово»
  if(step!=pstep || dom!=pdom){ pstep=step; pdom=dom; pmax=0; ts=tfr; pe=0; ph_f=ph_h=ph_g=ph_b=0 }
  if(tfr<ts) ts=tfr                                                            # часы отмотало назад — подстраховка от залипания
  # Ровный ход, пока шаг укладывается в среднюю длительность; дальше — всё медленнее, к концу подползаем,
  # но не доходим: иначе на шаге длиннее средней (после оценки, после сброса) полоса замирает и выглядит зависшей.
  if(sp>0 && ts){ pe=(tfr>ts?(tfr-ts)/sp:0); if(pe>0.9) pe=0.9+0.095*(1-exp(-(pe-0.9))); pp=pe }
  else pp=(ph_f+ph_h+ph_g+ph_b)/160   # длительность шага ещё не известна — грубо, по этапам
  if(pp>pmax) pmax=pp; pp=pmax
  if(dbg!="") printf "шаг %d часы %.3f начало %.3f длит %.3f полоса %.4f этапы %d/%d/%d/%d\n", step, tfr, ts, sp, pp, ph_f, ph_h, ph_g, ph_b >> dbg   # PANEL_DEBUG=файл
  ph=(ph_b>0?"Обратный проход · Adam":ph_g>0?"Градиент LM head":ph_h>0?"LM head":"Прямой проход")
  k=(n<every?n:every); a=0; aa=0; for(i=0;i<k;i++){ a+=q[i]; aa+=qa[i] } if(k){a/=k; aa/=k}
  F=""
  put("eb",1,10,bw,bar((step-1)/tot,"\033[1;96m")); put("ec",1,12+bw,28,B(sprintf("%d/%d",step-1,tot)) D(sprintf("  %.1f%%",100*(step-1)/tot)))
  if(n==0 && lp>0){ put("sb",2,10,bw,bar(lp,"\033[1;95m")); put("sp",2,12+bw,28,D("Загрузка модели")) }
  else { put("sb",2,10,bw,bar(pp,"\033[1;95m")); put("sp",2,12+bw,28,D(substr(ph,1,28))) }
  cellv(0,sprintf("%.3f",l)); cellv(1,sprintf("%.3f",a)); cellv(2,sprintf("%.1f",exp(l))); cellv(3,sprintf("%.0f%%",100*aa))
  cellv(4,sprintf("%.2e",gn)); cellv(5,sprintf("%.2e",gm)); cellv(6,lr); cellv(7,(mn==""?"—":sprintf("%.3f",mn)))
  cellv(8,hm(el)); cellv(9,(eta<0?"—":hm(eta))); cellv(10,(sp>0?sprintf("%.1f с",sp):"—")); cellv(11,(Tm>0?sprintf("%.1f",TKm/Tm):"—"))   # именно Tm, а не sp: скользящая оценка известна уже после первого шага, а сумма — только со второго
  cellv(12,(v==""?"—":v)); cellv(13,(tc==""?"—":tc"°C")); cellv(14,(fn==""?"—":fn)); cellv(15,(le==""?"—":le) (lacc!=""?sprintf(" %.0f%%",lacc):"") (be!=""&&be!=le?" / "be:"")); cellv(16,(ld!=""?ld:disk()))
  if(F!=""){ printf "%s", F > "/dev/stderr"; fflush("/dev/stderr") }
  if(el-elw>=1){ elw=el; ef="/tmp/" nm "_elapsed_now"; print el > ef; close(ef) } }   # время на диск — раз в секунду, а не каждый кадр
function hms(x){ x=int(x); return sprintf("%d:%02d:%02d",x/3600,(x%3600)/60,x%60) }
function emit(ev,  i,o){ c[1]=strftime("%Y-%m-%d %H:%M:%S"); c[2]=ev; o=c[1]; for(i=2;i<=38;i++) o=o "\t" c[i]; print o >> lt; fflush(lt); delete c }
function msg(ev,t){ c[3]=s+n-1; c[38]=t; emit(ev) }
BEGIN{ RS="[\r\n]+" }   # полосы программы идут через \r — делим и по ним, чтобы текст ошибки после них не потерялся
# Движок пишет ход шага строками с возвратом каретки без перевода строки, и кадр, напечатанный сразу за такой
# строкой, слипается с ней в одну запись. Поэтому время кадра вынимаем из любого места записи, а не только с начала —
# иначе терялось почти всё, и полоса шага шла грубыми ступенями по этапам.
{ tk=0
  while(match($0,/tick [0-9]+(\.[0-9]+)?/)){ tnow=substr($0,RSTART+5,RLENGTH-5)+0; $0=substr($0,1,RSTART-1) substr($0,RSTART+RLENGTH); tk=1 }
  if(tk && $0 ~ /^[ \t]*$/){ draw(); next } }
/Загрузка [0-9]/{ ld=$0; sub(/^.*Загрузка /,"",ld); lf=lt; sub(/_log\.tsv$/,"_load.txt",lf); print ld > lf; close(lf)          # ход чтения модели — в ячейку диска
  if(match(ld,/^[0-9.]+\/[0-9.]+/)){ split(substr(ld,RSTART,RLENGTH),gg,"/"); lp=(gg[2]>0? gg[1]/gg[2] : 0) }  # и в полосу: иначе на экране ничего не движется
  draw(); next }   # ход чтения модели — в ячейку диска, пока не пошли шаги
/^tick/{ if($2!="") tnow=$2+0; draw(); next }
/^prog f/{ ph_f=$3; ph_h=ph_g=ph_b=0; next }   # ts — начало шага (для плавной полосы)
/^prog h/{ ph_h=$3; next }
/^prog g/{ ph_g=$3; next }
/^prog b/{ ph_b=$3; next }
/forward layer [0-9]+\/[0-9]+ done|backward\+Adam layer|LM head backward chunk|^d\(final_norm_input\)|^[ \t]*$/{ next }   # счётчики хода — не события
/^=== EVAL step/{ match($0,/mean loss [0-9.]+/); le=substr($0,RSTART+10,RLENGTH-10); if(be==""||le+0<be+0) be=le; match($0,/step [0-9]+/); es=substr($0,RSTART+5,RLENGTH-5)
  match($0,/[0-9]+ answer tokens/); c[8]=substr($0,RSTART,RLENGTH-14); match($0,/[0-9]+ packs/); c[6]=substr($0,RSTART,RLENGTH-6) " упаковок"
  match($0,/acc [0-9.]+/); lacc=(RSTART?substr($0,RSTART+4,RLENGTH-4)*100:"")
  c[3]=es; c[14]=sprintf("%.4f",le); c[15]=sprintf("%.4f",be); c[10]=sprintf("%.2f",exp(le)); if(lacc!="") c[11]=sprintf("%.1f",lacc); c[24]=hms(st0+systime()-t0); emit("Оценка"); draw(); next }
/^Лучший снимок весов:/{ msg("Лучший снимок",$0); next }
/^Снимок весов сохранён:/{ msg("Сохранение снимка",$0); next }
/^mem live/{ mlv=$3; mlb=substr($5,2); mpl=$8; mms=$9; gsub(/[()]/,"",mms); next }
/forward loss/{ l=$4; match($0,/ex -?[0-9]+/); ex=substr($0,RSTART+3,RLENGTH-3); match($0,/n_ex [0-9]+/); nx=(RSTART?substr($0,RSTART+5,RLENGTH-5):1); match($0,/len [0-9]+/); ln=substr($0,RSTART+4,RLENGTH-4); match($0,/loss on [0-9]+/); nl=substr($0,RSTART+8,RLENGTH-8)
  match($0,/acc [0-9.]+/); ac=substr($0,RSTART+4,RLENGTH-4)+0; q[n%every]=l; qa[n%every]=ac; C++; W+=l; draw(); next }
/--- step/{ if(ld!=""){ lf=lt; sub(/_log\.tsv$/,"_load.txt",lf); printf "" > lf; close(lf) } ld=""; match($0,/done \[[0-9.]+s/); stt=substr($0,RSTART+6,RLENGTH-7)+0; T+=stt; TK+=ln; n++; if(n>1){ Tm+=stt; TKm+=ln; m++ }
  # скользящая оценка длительности шага: простое среднее тянет за собой редкие затяжные шаги (оценка, сброс на диск)
  # до конца прогона, поэтому выбросы втрое выше текущей оценки в неё не берём, а остальное усредняем с весом 0.3
  if(spe==0) spe=stt; else if(stt<3*spe) spe=0.7*spe+0.3*stt
  match($0,/gnorm [0-9.e+-]+/); gn=substr($0,RSTART+6,RLENGTH-6)+0; G+=gn; if(gn>gm) gm=gn
  match($0,/VRAM peak [0-9]+\/[0-9]+/); v=substr($0,RSTART+10,RLENGTH-10); match($0,/mem [0-9\/]+ C/); tc=substr($0,RSTART+4,RLENGTH-6); match($0,/fan max [0-9]+/); fn=substr($0,RSTART+8,RLENGTH-8)
  k=(n<every?n:every); a=0; aa=0; for(i=0;i<k;i++){ a+=q[i]; aa+=qa[i] } a/=k; aa/=k; if(n>=every && (mn==""||a<mn)) mn=a
  match($0,/RAM cache [0-9.]+/); rc=substr($0,RSTART+10,RLENGTH-10); match($0,/resident [0-9]+/); rs=substr($0,RSTART+9,RLENGTH-9)
  pf=ph1=ph2=""; if(match($0,/phases [0-9.]+\/[0-9.]+\/[0-9.]+/)){ split(substr($0,RSTART+7,RLENGTH-7),pz,"/"); pf=pz[1]; ph1=pz[2]; ph2=pz[3] }
  split(v,vv,"/"); split(tc,tt,"/"); st=s+n-1; sp=(m>0?Tm/m:stt)
  c[3]=st; c[4]=int((st-1-base)/npe)+1; c[5]=ex; c[6]=nx; c[7]=ln; c[8]=nl; c[9]=sprintf("%.4f",l); c[10]=sprintf("%.2f",exp(l)); c[11]=sprintf("%.1f",100*ac)
  c[12]=sprintf("%.4f",a); c[13]=sprintf("%.1f",100*aa); c[15]=(be==""?"":sprintf("%.4f",be)); c[16]=sprintf("%.2f",gn); c[17]=sprintf("%.2f",gm); c[18]=lr
  c[19]=sprintf("%.2f",stt); c[20]=pf; c[21]=ph1; c[22]=ph2; c[23]=sprintf("%.1f",(stt>0?ln/stt:0)); c[24]=hms(st0+systime()-t0); c[25]=hms((tot-st)*sp)
  c[26]=vv[1]; c[27]=vv[2]; c[28]=mlv; c[29]=mlb; c[30]=mpl; c[31]=mms; c[32]=tt[1]; c[33]=tt[2]; c[34]=tt[3]; c[35]=fn; c[36]=rc; c[37]=rs
  emit("Шаг"); draw(); next }
{ msg("Сообщение",$0) }   # всё прочее от программы — сведения о запуске, предупреждения, ошибки HIP, утечки
END{ printf "\033[?25h\n" > "/dev/stderr"; if(C) printf "Шаги %d–%d: средняя потеря (loss) на обучении %.4f.\n", s, s+C-1, W/C; else { printf "Ошибка: обучающая программа завершилась без единого шага. Подробности: события «Сообщение» в журнале %s_log.tsv рабочей папки.\n", nm; exit 1 } }
AWK
)
AWK=$(command -v gawk || command -v awk) # ж1
mats_tier(){ awk -v spec="$MATS" 'BEGIN{ split(spec,a,":"); nk=split(a[1],K,"+"); n=split(a[2],L,",")
    for(j=1;j<=nk;j++){ ks=(K[j]=="attn" ? "q_proj k_proj v_proj o_proj" : K[j]); nn=split(ks,KK," ")
      for(q=1;q<=nn;q++) for(i=1;i<=n;i++) want["layer_" L[i] "." KK[q]]=1 } }
  /^#/{ print; next } { if($1 in want){ print $1, 1, $3, $3; tot+=$3; k++ } else print $1, 2, $3, 0 }
  END{ printf "Целые матрицы: %s, слои %s (матриц: %d), обучается %d чисел, остальное заморожено.\n", a[1], a[2], k, tot > "/dev/stderr"; exit (k?0:1) }' "$1" > "$1.t" && mv "$1.t" "$1"; }   # з1
run(){ $B $M /dev/null 8 "$@" 2>&1; }   # и1
if [ ! -f $TIER ] && [ -n "${LORA:-}" ]; then   # и2
  echo "# LoRA $LORA: исходные веса заморожены" > $TIER; rm -f $ADAM $CKPT $D/${NAME}_step.txt
elif [ ! -f $TIER ]; then   # и3
  stage Раскладка; echo "Построение раскладки среза"; lg Запуск "Построение раскладки среза"
  run /dev/null /dev/null $LR 9999 1 --retier $X --tier-file $TIER | grep -a 'Срез строк\|Раскладка среза' | while IFS= read -r x; do echo "$x"; lg Сообщение "$x"; done
  if [ -n "${MATS:-}" ]; then x=$(mats_tier $TIER 2>&1) || { echo "Матрицы не найдены: $MATS."; exit 1; }; echo "$x"; lg Сообщение "$x"; fi
  rm -f $ADAM $CKPT $D/${NAME}_step.txt
fi
[ -f $D/${NAME}_step.txt ] || rm -f $CKPT $ADAM $D/${NAME}_elapsed.txt   # и4
[ -f $CKPT ] || rm -f $D/${NAME}_step.txt $D/${NAME}_elapsed.txt $D/${NAME}_tier1_best.bin   # и5
S=$(cat $D/${NAME}_step.txt 2>/dev/null || echo 1)
[ $S -gt $TOTAL ] && { echo "=== Обучение уже завершено: $TOTAL шагов ==="; exit 0; }
EF=; [ "$S" = 1 ] && { echo "=== Параметры запуска: LR $LR, до $PACK токенов в шаге, $N шагов на эпоху × $EPOCHS = $TOTAL шагов, валидация каждые $EV шагов, снимок весов каждые $FL шагов, порядок упаковок меняется в каждой эпохе ===" ; lg Запуск "LR $LR, до $PACK токенов в шаге, $N шагов на эпоху × $EPOCHS = $TOTAL шагов, валидация каждые $EV шагов, снимок весов каждые $FL шагов"
  awk -F'\t' '$2=="Оценка"&&$3==0{f=1} END{exit !f}' $LT || EF=--eval-first; }
LE=$(awk -F'\t' '$2=="Оценка"{v=$14} END{print v}' $LT) || true; BE=$(awk -F'\t' '$2=="Оценка"&&(m==""||$14<m){m=$14} END{print m}' $LT) || true
stage Обучение
# и6
( run $ADAM $CKPT $LR 9999 $S $X --tier-file $TIER --data $TOK --steps $((TOTAL-S+1)) --progress --pack $PACK --shuffle \
      --eval-data $D/${NAME}_eval.bin --eval-every $EV $EF --stop-after $STOP_AFTER --flush-every $FL --step-file $D/${NAME}_step.txt --best $D/${NAME}_tier1_best.bin & P=$!
  FIFO=$(mktemp -u /tmp/${NAME}_tick.XXXX); mkfifo "$FIFO" 2>/dev/null && exec 9<>"$FIFO" && rm -f "$FIFO"   # и7
  DT=$(LC_ALL=C awk -v f="${FPS:-60}" 'BEGIN{printf "%.4f", 1/f}')   # и8
  # и9
  ( trap '' PIPE
    while :; do t=${EPOCHREALTIME:-}; t=${t//,/.}; [ -n "$t" ] || t=$(date +%s.%N)
      printf 'tick %s\n' "$t" || break
      if [ -e /dev/fd/9 ]; then read -t $DT -u 9 _ 2>/dev/null; else sleep $DT; fi; done ) & TP=$!
  wait $P && RC=0 || RC=$?; kill $TP 2>/dev/null || true; wait $TP 2>/dev/null || true; if [ $RC -eq 75 ]; then printf 'Обучение остановлено по потолку памяти, снимок весов записан. Для продолжения требуется освободить память и повторить запуск: обучение продолжится с этого шага.\n' >&2; elif [ $RC -ne 0 ]; then printf 'Движок завершился с ошибкой (код %s). Подробности: sudo dmesg | tail\n' "$RC" >&2; fi; exit $RC ) \
 | LC_ALL=C.UTF-8 $AWK -v nm="$NAME" -v ttl="$TITLE" -v s=$S -v tot=$TOTAL -v every=${EVERY:-50} -v lt=$LT -v base=$BASE -v npe=$N -v lr=$LR -v le="$LE" -v be="$BE" -v t0=$T0 -v cols=$(tput cols 2>/dev/null || echo 100) -v dev=$(lsblk -no PKNAME $(findmnt -no SOURCE $D) 2>/dev/null | head -1) -v st0=$(cat $D/${NAME}_elapsed.txt 2>/dev/null || echo 0) -v dbg="${PANEL_DEBUG:-}" "$DASH"
cat /tmp/${NAME}_elapsed_now 2>/dev/null > $D/${NAME}_elapsed.txt || true
DONE=$(awk -F'\t' '$2=="Шаг"{v=$3} END{print v}' $LT) || true   # и10
[ "${DONE:-$TOTAL}" = "$TOTAL" ] && W="$TOTAL шагов" || W="$DONE из $TOTAL шагов · ранняя остановка"
stage Завершено; echo "=== Обучение завершено: $W · лучший снимок весов: $D/${NAME}_tier1_best.bin ==="; lg Завершено "$W · Лучший снимок весов: $D/${NAME}_tier1_best.bin"
