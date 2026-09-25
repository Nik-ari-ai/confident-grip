import fs from 'node:fs/promises';
import path from 'node:path';
import {Presentation, PresentationFile} from '@oai/artifact-tool';
const root=path.resolve(process.env.PROJECT_ROOT || '.');
const p=Presentation.create({slideSize:{width:1280,height:720}});
const green='#28533A', black='#232A25', gray='#59635B', pale='#ECF1EB';
const font='Arial';
const percent=x=>(x*100).toFixed(1).replace('.',',')+'%';
function text(s,value,left,top,width,height,size=28,color=black,bold=false){
 const sh=s.shapes.add({geometry:'textbox',position:{left,top,width,height},fill:'none',line:{fill:'none',width:0}});
 sh.text=value;sh.text.style={typeface:font,fontSize:size,color,bold,autoFit:'none'};return sh;
}
function slide(title,n){const s=p.slides.add();s.background.fill='#FFFFFF';text(s,title,64,40,1150,66,42,green,true);text(s,String(n),1170,666,45,24,17,gray);return s;}
function table(s,values,left,top,width,height,widths){
 const tb=s.tables.add({rows:values.length,columns:values[0].length,left,top,width,height,columnWidths:widths,values});
 tb.borders.assign({fill:'#D1DAD2',width:.7,style:'solid'});
 tb.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({
  fill:'#FFFFFF',textStyle:{typeface:font,fontSize:23,color:black},margins:{left:14,right:12,top:10,bottom:10}});
 tb.cells.block({row:0,column:0,rowCount:1,columnCount:values[0].length}).assign({fill:pale,textStyle:{typeface:font,fontSize:23,color:green,bold:true}});
 return tb;
}
function notes(s,t){s.speakerNotes.textFrame.setText(t);}
const report2=JSON.parse(await fs.readFile(path.join(root,'results/study/s2/metrics.json'),'utf8'));
const report3=JSON.parse(await fs.readFile(path.join(root,'results/study/s3/metrics.json'),'utf8'));
let s=slide('Уверенный хват',1);
text(s,'Как не менять хват\nиз-за сомнительной команды',64,180,1110,135,47,black,true);
text(s,'Пять хватов и покой по поверхностной ЭМГ.\nПрограммный прототип и проверка на новых записях.',64,365,1100,115,30,gray);
text(s,'SK Challenge 2026 · студенческий кейс СТЕПЛАЙФ\n25 сентября 2026',64,566,1100,76,24,gray);
notes(s,'Проверяется компромисс ошибок, отказов и задержки. Это программный прототип; кисть не подключена. Ограничение полной реакции 100 мс пока не подтверждено.');

s=slide('Что считываем и как проверяем',2);
text(s,'8 каналов предплечья · 2 кГц\nNinaPro DB2',64,142,535,95,32,black,true);
text(s,'S1 — выбор признаков и окна.\nS2 и S3 — новые участники\nдля проверки процедуры.',64,281,535,128,29);
text(s,'У каждого своя настройка:\ntrain 1, 3, 4 · validation 2 · test 5, 6.\nЭто не перенос модели между людьми.',64,475,545,132,25,gray);
table(s,[['Хват','Метка DB2'],['Цилиндрический','19'],['Точный','32'],['Боковой','34'],['Крюк','20'],['Сферический','27'],['Покой','0']],650,141,565,457,[370,195]);
notes(s,'Источник данных: https://ninapro.hevs.ch/instructions/DB2.html ; таблица движений: https://ninapro.hevs.ch/figures/SData_Movements.png . Первые восемь каналов. Все участники без ампутации. На каждого приходится 30 выбранных фрагментов: пять движений по шесть повторов. Включено до 0,5 с соседнего покоя.');

s=slide('Обработка и проверка команды',3);
const topNodes=['Электроды','Усилитель\nи АЦП','Причинный\nфильтр','Окно\n100/150 мс','Признаки\n40 значений'];
const bottomNodes=['Начало\nдвижения','Контроллер\nпривода','Связь','Проверка\nкоманды','LDA / SVM'];
for(let row=0;row<2;row++)for(let i=0;i<5;i++){
 const x=64+i*235,y=row?315:155;
 s.shapes.add({geometry:'rect',position:{left:x,top:y,width:205,height:88},fill:row===1&&i===3?pale:'#FFFFFF',line:{fill:green,width:1.5}});
 text(s,(row?bottomNodes:topNodes)[i],x+12,y+13,181,65,25,black);
 if(i<4)text(s,row?'←':'→',x+205,y+25,42,44,18,green);
}
text(s,'↓',1117,254,50,45,28,green);
text(s,'Оценка модели + разница двух лучших оценок + подтверждение.\nПлохой сигнал или отказ → запрет движения.',64,459,1150,108,28);
text(s,'Проверка не исправляет уверенный неправильный ответ.\nНа приводе нужен независимый таймер отсутствия команд.',64,596,1140,66,23,gray);
notes(s,'Реализован участок от цифрового сигнала до команды. Фильтр 20–450 Гц и notch 50 Гц, только прошлые отсчёты. RMS, MAV, WL, ZC, SSC. В SVM — логарифмирование, окно 100 мс; LDA — исходные признаки, окно 150 мс. Оценки калибруются внутри train по повторам. Покой не означает раскрытие кисти.');

s=slide('Две модели, два новых участника',4);
table(s,[['Модель','S2: точность / F1','S3: точность / F1'],...['LDA','SVM'].map(n=>[n,...[report2,report3].map(r=>{const m=r.models[n].test.baseline;return percent(m.raw_accuracy)+' / '+m.raw_macro_f1.toFixed(3)})])],64,145,1150,239,[280,435,435]);
text(s,'На S1 сравнены 24 конфигурации. Выбор закреплён до теста.\nLDA: обычные признаки, 150 мс. SVM: логарифмы, 100 мс.',64,425,1140,95,28);
text(s,'SVM на одинаковых временных точках, старая → новая версия:\nS2: 61,7 → 65,2%     S3: 52,0 → 57,7%',64,565,1140,87,27,green,true);
notes(s,'Протокол и конфигурации: коммит fe22ce7, docs/protocol-v2.md. Выбирается macro-F1 на валидации S1. Окна 50,75,100,150 мс; td/log_td; LDA и SVM C=1/10. Сравнение старой и новой версии на совпадающих временных точках: results/study/paired_comparison.json. Вклад отдельных правок не изолирован.');

s=slide('Меньше переключений — какой ценой?',5);
table(s,[['SVM','S2','S3'],['Ошибочные переключения/мин: до → после','482 → 178','546 → 175'],['Принято после проверки','86,5%','67,8%'],['Ошибки среди принятых','29,3%','32,0%'],['Полнота распознавания покоя','14,7%','15,9%']],64,140,1150,340,[660,245,245]);
text(s,'Режим выбирался при ≥70% принятых окон на валидации.\nНа тесте S3 покрытие упало ниже этого ограничения.',64,522,1150,83,27);
text(s,'Слабое место — распознавание покоя. Для настоящей кисти ошибок слишком много.',64,629,1120,30,21,gray);
notes(s,'Для S2 выбраны три подтверждения без порога. S3: threshold=0.5, margin=0.15, confirmations=2. Валидационное покрытие S3 70,96%, тестовое 67,84%. На обоих участниках получена правильная команда в каждом из десяти движений, но ранний правильный кадр не равен устойчивому управлению. Ошибочное переключение — смена принятого класса на неверный.');

s=slide('Вычисления быстрые. Ответ пока нестабилен.',6);
table(s,[['SVM после проверки','S2','S3'],['Вычислительная итерация, p95','1,16 мс','1,29 мс'],['Первая правильная команда за 100 мс','6 из 10','6 из 10'],['Медиана устойчивого ответа','375 мс','980 мс']],64,143,1150,282,[660,245,245]);
text(s,'Устойчивый ответ: закончились 100 мс подряд правильных команд.\nПолная задержка до движения кисти не измерена.',64,464,1150,90,27);
text(s,'Для следующего стенда: проектный бюджет 92 мс + 8 мс резерва.\nОн требует нового варианта с окном 40 мс и проверки качества.',64,587,1150,80,25,gray);
notes(s,'Отсчёт по restimulus, не по независимому измерению сокращения. Нулевая медиана первого ответа S2 не является мгновенной физической реакцией. Сбор информативного участка 40 + шаг10 + задержка тракта5 + признаки1 + модель3 + подтверждение10 + связь3 + движение20 =92 мс. Это лимиты проектирования, не измеренные показатели. Выбранные окна 100/150 мс не подтверждают выполнение ТЗ. Подробнее: отчёт, страницы 5–6.');

s=slide('Что нужно для аппаратной версии',7);
text(s,'Сначала: покой и короткие окна\nна новом отложенном тесте.\n\nЗатем: съём, калибровка, плата\nи проверка с приводом.\n\nПлан данных: 8–12 пользователей,\n5 хватов × 10 повторов × 3 визита.',64,145,580,373,28);
table(s,[['Стенд','Лимит, руб.'],['Съём и электроды','47 000'],['STM32 и интерфейсы','10 000'],['Приводы и механика','65 000'],['Питание и сборка','20 000'],['Итерации и расходники','18 000'],['Итого','160 000']],682,141,532,409,[362,170]);
text(s,'3–6 месяцев. Сумма — распределение бюджета, цены ещё проверяются.\nВ комплекте: код, журнал эксперимента и покадровое демо.',64,593,1140,85,25,gray);
notes(s,'Съём ADS1298, вычислитель STM32H743: https://www.ti.com/product/ADS1298 ; https://www.st.com/en/microcontrollers-microprocessors/stm32h743-753.html . Резерв до 200000 руб. —40000. Нужна лаборатория; зарплата и клинические испытания не включены. docs/demo.html воспроизводит сохранённые результаты; не запускает модель и не управляет кистью. Само отклонение решений не заявляется новой идеей: https://arxiv.org/abs/2409.14169 .');


const output=path.join(root,'docs/presentation.pptx');
await(await PresentationFile.exportPptx(p)).save(output);
console.log(output);
