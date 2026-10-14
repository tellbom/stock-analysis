import fs from 'node:fs/promises';
import {Workbook,SpreadsheetFile,FileBlob} from '@oai/artifact-tool';
const dirIndex=process.argv.indexOf('--dir');
const dir=dirIndex>=0?process.argv[dirIndex+1]:'E:/stock-analysis/outputs/01a116ce-1f0c-7d41-8967-adcca880c1b8/原版统一基线对照_20260930';
const out=`${dir}/普通与混合D3_D5_统一基线对照.xlsx`;
if(process.argv.includes('--render')){
 const w=await SpreadsheetFile.importXlsx(await FileBlob.load(out));
 for(const [sheetName,range] of [['对照结果','A1:I35'],['训练口径','A1:G20'],['逐股收益','A1:L16'],['行情输入','A1:H16']]){
  const png=await w.render({sheetName,range,scale:1.2,format:'png'});
  await fs.writeFile(`${dir}/${sheetName}.png`,new Uint8Array(await png.arrayBuffer()));
 }
 process.exit(0);
}
const x=JSON.parse(await fs.readFile(`${dir}/workbook_input.json`,'utf8'));
const w=Workbook.create(),s=w.worksheets.add('对照结果'),t=w.worksheets.add('训练口径'),d=w.worksheets.add('逐股收益'),p=w.worksheets.add('行情输入');
const font={name:'Microsoft YaHei',size:10,color:'#24364B'},pct='0.00%;[Red]-0.00%',date=v=>new Date(`${v}T00:00:00Z`);
function setup(sheet,last,end){sheet.showGridLines=false;sheet.getRange(`A1:${last}${end}`).format={font,rowHeight:24,verticalAlignment:'center'};sheet.getRange('A2').format.font={...font,bold:true,size:14};}
function merged(sheet,row,last,text){sheet.getRange(`A${row}:${last}${row}`).merge();sheet.getRange(`A${row}`).values=[[text]];}
function header(sheet,row,last,values){sheet.getRange(`A${row}:${last}${row}`).values=[values];sheet.getRange(`A${row}:${last}${row}`).format={fill:'#244664',font:{...font,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',rowHeight:28};}
function widths(sheet,values){values.forEach((v,i)=>sheet.getRange(`${String.fromCharCode(65+i)}:${String.fromCharCode(65+i)}`).format.columnWidth=v);}
setup(s,'I',35);setup(t,'G',60);setup(d,'L',x.details.length+8);setup(p,'H',x.prices.length+7);
merged(s,2,'I','普通与混合 D3 / D5：统一原版基线对照');
merged(s,4,'I','回溯实验：信号 9/30，10/8 收盘入场，收益观察至 10/13。混合名单为本次事后重建。');
s.getRange('A6:C7').values=[['信号日期',date(x.train.as_of),null],['入场日期',date(x.train.entry_date),null]];
s.getRange('D6:F7').values=[['观察日期',date(x.performance.observed_through),null],['每边成本',x.performance.one_way_cost_bps/10000,null]];
s.getRange('G6:I7').values=[['D3等量只数',x.train.horizons['3'].equal_count_n,null],['D5等量只数',x.train.horizons['5'].equal_count_n,null]];
s.getRange('B6:B7').setNumberFormat('yyyy-mm-dd');s.getRange('E6').setNumberFormat('yyyy-mm-dd');s.getRange('E7').setNumberFormat(pct);
header(s,9,'I',['周期','同数量对照策略','股票数','退出净收益','沪深300','净超额收益','较原版变化','个股上涨率','持有期状态']);
header(s,23,'I',['周期','各组完整名单参考','股票数','退出净收益','沪深300','净超额收益','较原版变化','个股上涨率','持有期状态']);
merged(s,19,'I','等量规则：先按行业最多3只、总数最多50只选股，再取四组最少只数；只数在读取收益前确定。');
merged(s,20,'I','D3 已于 10/13 到期；D5 于 10/15 到期，本表 D5 是假设 10/13 提前卖出的阶段净收益。');
merged(s,21,'I','原版完整名单仍为50只；上方D3为49只等量对照，下方保留50只原版收益。');
merged(s,33,'I','当前评审意见：保留原版为固定基线；混合留作实验组，证据不足以替换原版。两组均未跑赢指数。');
merged(s,34,'I','D3资金流混合改善收益但上涨率未提高；分数Gate降低收益。D5尚未到期，不能判定完整表现。');
merged(d,2,'L','逐股收益：全部选中记录与等量比较范围');
merged(d,4,'L','本次回溯实验使用同一份原版Ridge32预测。行业最多3只、总数最多50只；等量组取名单前N只。');
merged(d,5,'L','净收益 = 毛收益 - 双边成本；D3/D5观察截止均为10/13，D5属于提前退出假设。');
header(d,8,'L',['周期','策略','等量组内','名次','股票代码','名称','行业','入场收盘','观察收盘','毛收益','退出净收益','净超额收益']);
merged(p,2,'H','行情输入与基准收益');
merged(p,4,'H','推荐来源：9/30原交付Ridge预测；混合实验由 compare_original_hybrid.py 重建，并未在9/30实际交付。');
merged(p,5,'H','价格来源：腾讯日线前复权请求；仅10/8、9、12、13四个交易日。API见下方。');
merged(p,6,'H','https://web.ifzq.gtimg.cn/appstock/app/fqkline/get');
header(p,7,'D',['股票代码','交易日期','收盘价','行情来源']);
const pe=x.prices.length+7;
p.getRange(`A8:D${pe}`).values=x.prices.map(r=>[r.symbol,date(r.date),r.close,r.source]);
p.getRange(`A8:A${pe}`).setNumberFormat('@');p.getRange(`B8:B${pe}`).setNumberFormat('yyyy-mm-dd');p.getRange(`C8:C${pe}`).setNumberFormat('0.00');
const pa=`'行情输入'!$A$8:$A$${pe}`,pb=`'行情输入'!$B$8:$B$${pe}`,pc=`'行情输入'!$C$8:$C$${pe}`;
const lookup=(symbol,day)=>`IF(COUNTIFS(${pa},${symbol},${pb},${day})=1,SUMIFS(${pc},${pa},${symbol},${pb},${day}),"n.a.")`;
p.getRange('F7:H7').values=[['沪深300入场','沪深300观察','基准收益']];
p.getRange('F8:H8').formulas=[[`=${lookup('"000300"',"'对照结果'!$B$7")}`,`=${lookup('"000300"',"'对照结果'!$E$6")}`,'=G8/F8-1']];p.getRange('H8').setNumberFormat(pct);
const de=x.details.length+8;
d.getRange(`A9:L${de}`).values=x.details.map(r=>[`D+${r.horizon}`,r.strategy_name,r.rank<=x.train.horizons[String(r.horizon)].equal_count_n?'是':'否',r.rank,r.symbol,r.name,r.industry_name,null,null,null,null,null]);
d.getRange(`H9:L${de}`).formulas=x.details.map((r,i)=>{const n=i+9;return [
 `=${lookup(`E${n}`,"'对照结果'!$B$7")}`,`=${lookup(`E${n}`,"'对照结果'!$E$6")}`,
 `=IF(AND(ISNUMBER(H${n}),ISNUMBER(I${n}),H${n}>0),I${n}/H${n}-1,"n.a.")`,
 `=IF(ISNUMBER(J${n}),J${n}-2*'对照结果'!$E$7,"n.a.")`,
 `=IF(ISNUMBER(K${n}),K${n}-'行情输入'!$H$8,"n.a.")`];});
d.getRange(`E9:E${de}`).setNumberFormat('@');d.getRange(`H9:I${de}`).setNumberFormat('0.00');d.getRange(`J9:L${de}`).setNumberFormat(pct);
const da=`'逐股收益'!$A$9:$A$${de}`,db=`'逐股收益'!$B$9:$B$${de}`,dc=`'逐股收益'!$C$9:$C$${de}`,dj=`'逐股收益'!$J$9:$J$${de}`,dk=`'逐股收益'!$K$9:$K$${de}`;
const equal=x.performance.strategies.filter(r=>r.scope==='equal_count'),full=x.performance.strategies.filter(r=>r.scope==='full_selection');
for(const [scope,start,records] of [['equal_count',10,equal],['full_selection',24,full]]){
 records.forEach((r,i)=>{
  const n=start+i,criteria=`${da},A${n},${db},B${n}`+(scope==='equal_count'?`,${dc},"是"`:''),baseRow=start+(r.horizon===3?0:4);
  s.getRange(`A${n}:I${n}`).values=[[`D+${r.horizon}`,r.strategy_name,null,null,null,null,null,null,r.status]];
  s.getRange(`C${n}:H${n}`).formulas=[[
   `=COUNTIFS(${criteria},${dk},">-1E99")`,
   `=IF(C${n}=${r.count},AVERAGEIFS(${dk},${criteria}),"n.a.")`,
   `='行情输入'!$H$8`,
   `=IF(ISNUMBER(D${n}),D${n}-E${n},"n.a.")`,
   `=IF(ISNUMBER(D${n}),D${n}-D${baseRow},"n.a.")`,
   `=IF(C${n}=${r.count},COUNTIFS(${criteria},${dj},">0")/C${n},"n.a.")`]];
  if(i<4)s.getRange(`A${n}:I${n}`).format.fill='#EAF1F7';
 });
}
s.getRange('D10:H17').setNumberFormat(pct);s.getRange('D24:H31').setNumberFormat(pct);
merged(t,2,'G','训练口径：原始模型固定，资金流只加入近期模型');
merged(t,4,'G','数据源：training_summary.json、D3/D5近期重要性CSV与原交付JSON。原版50只顺序与分数已核验。');
merged(t,5,'G','D3和D5独立训练；RRF固定k=60，不训练融合权重。新实验Gate只检查分数层级，不增加事件因子。');
header(t,7,'G',['周期','模型','因子数','训练起点','最后训练样本','样本数','训练日数']);
const trains=[];
for(const h of [3,5]){const z=x.train.horizons[String(h)];trains.push(
 [`D+${h}`,'原版Ridge（各混合共用）',32,date(z.original_train_start),date(z.original_train_end),z.original_train_rows,'长期全量'],
 [`D+${h}`,'近期LightGBM无资金流',32,date(z.recent_train_start),date(z.recent_train_end),z.shared_recent_train_rows,z.shared_recent_dates],
 [`D+${h}`,'近期LightGBM有资金流',37,date(z.recent_train_start),date(z.recent_train_end),z.shared_recent_train_rows,z.shared_recent_dates]);}
t.getRange('A8:G13').values=trains;t.getRange('D8:E13').setNumberFormat('yyyy-mm-dd');t.getRange('F8:F13').setNumberFormat('#,##0');
merged(t,15,'G','两近期组样本、日期、参数及股票池相同，仅有资金流组增加下方5列；缺失资金流排除行数为0。');
merged(t,16,'G','原版重建分数最大差异小于2×10⁻¹⁵，50只顺序相同；各混合共用同一份299只原版预测。');
merged(t,17,'G','五个资金流因子都有120个完整训练日，且实际用于树分裂；分裂次数不能证明提高选股准确性。');
merged(t,18,'G','模型已另存joblib作为本次实验审计证据；未更新或晋升注册的生产模型。');
merged(t,19,'G','资金流历史为事后补齐数据；截至信号日截断，但尚未独立证实供应商修订与当时历史可得性。');
header(t,21,'G',['因子名称','原版Ridge','近期无流','近期有流','因子类型','D3分裂次数','D5分裂次数']);
const features=[...x.train.original_features,...x.train.flow_features];
t.getRange(`A22:G${features.length+21}`).values=features.map(f=>{const flow=x.train.flow_features.includes(f);return [f,flow?'否':'是',flow?'否':'是','是',flow?'资金流':'原版因子',x.importances['3'][f],x.importances['5'][f]];});
widths(s,[8,36,10,14,14,14,14,13,24]);widths(t,[32,30,14,18,20,15,15]);widths(d,[8,36,13,8,12,17,24,14,14,14,14,15]);widths(p,[12,18,14,35,4,20,20,16]);
s.freezePanes.freezeRows(9);d.freezePanes.freezeRows(8);p.freezePanes.freezeRows(7);t.freezePanes.freezeRows(21);
w.recalculate();
for(const [start,records] of [[10,equal],[24,full]])records.forEach((r,i)=>{const n=start+i;if(s.getRange(`C${n}`).values[0][0]!==r.count)throw Error('Comparison count mismatch');for(const [c,k] of [['D','net_if_exit'],['E','benchmark_return'],['F','excess_if_exit'],['G','return_delta_vs_ordinary'],['H','stock_win_rate']])if(Math.abs(s.getRange(`${c}${n}`).values[0][0]-r[k])>1e-10)throw Error(`${c}${n} mismatch`);});
// Verify linked price/cost edits recalculate, restoring the saved inputs.
const oldNet=s.getRange('D10').values[0][0],oldCost=s.getRange('E7').values[0][0];
s.getRange('E7').values=[[oldCost+0.001]];w.recalculate();
if(Math.abs(s.getRange('D10').values[0][0]-(oldNet-0.002))>1e-10)throw Error('Cost edit did not recalculate');
s.getRange('E7').values=[[oldCost]];w.recalculate();
const pi=x.prices.findIndex(r=>r.symbol===x.details[0].symbol&&r.date===x.performance.observed_through)+8,oldPrice=p.getRange(`C${pi}`).values[0][0],oldStock=d.getRange('J9').values[0][0];
p.getRange(`C${pi}`).values=[[oldPrice*1.01]];w.recalculate();
if(Math.abs(d.getRange('J9').values[0][0]-((oldStock+1)*1.01-1))>1e-10)throw Error('Price edit did not recalculate');
p.getRange(`C${pi}`).values=[[oldPrice]];w.recalculate();
console.log((await w.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!',options:{useRegex:true,maxResults:10},maxChars:600})).ndjson);
await(await SpreadsheetFile.exportXlsx(w)).save(out);console.log(out);
