import fs from 'node:fs/promises';
import {Workbook,SpreadsheetFile,FileBlob} from '@oai/artifact-tool';
const dir=process.env.STOCK_WEEKLY_OUTPUT_DIR || 'E:/stock-analysis/outputs/01a116ce-1f0c-7d41-8967-adcca880c1b8/2026-10-14_复权修复_D3_D5';
const out=`${dir}/2026-10-14_D3_D5_复权修复_普通与混合.xlsx`;
if(process.argv.includes('--render')){
 const w=await SpreadsheetFile.importXlsx(await FileBlob.load(out));
 for(const [sheetName,range] of [['本轮口径','A1:H38'],['D3选股对照','A1:O18'],['D5选股对照','A1:O18'],['Gate诊断','A1:I16'],['训练因子','A1:G18'],['行情与覆盖','A1:K16']]){
  const b=await w.render({sheetName,range,scale:1.2,format:'png'});await fs.writeFile(`${dir}/${sheetName}.png`,new Uint8Array(await b.arrayBuffer()));
 }
 process.exit(0);
}
const x=JSON.parse(await fs.readFile(`${dir}/workbook_input.json`,'utf8')),m=x.summary,w=Workbook.create();
const s=w.worksheets.add('本轮口径'),views={'3':w.worksheets.add('D3选股对照'),'5':w.worksheets.add('D5选股对照')},g=w.worksheets.add('Gate诊断'),f=w.worksheets.add('训练因子'),p=w.worksheets.add('行情与覆盖');
const font={name:'Microsoft YaHei',size:10,color:'#24364B'},navy='#244664',pct='0.00%;[Red]-0.00%',date=v=>new Date(`${v}T00:00:00Z`);
const rowStats=[];
function setup(sheet,last,end){sheet.showGridLines=false;sheet.getRange(`A1:${last}${end}`).format={font,rowHeight:24,verticalAlignment:'center'};sheet.getRange('A2').format.font={...font,size:14,bold:true,color:navy};}
function merged(sheet,row,last,text){sheet.getRange(`A${row}:${last}${row}`).merge();sheet.getRange(`A${row}`).values=[[text]];}
function header(sheet,row,last,values){sheet.getRange(`A${row}:${last}${row}`).values=[values];sheet.getRange(`A${row}:${last}${row}`).format={fill:navy,font:{...font,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',rowHeight:28};}
function widths(sheet,vals){vals.forEach((v,i)=>sheet.getRange(`${String.fromCharCode(65+i)}:${String.fromCharCode(65+i)}`).format.columnWidth=v);}
setup(s,'H',39);merged(s,2,'H','10月14日复权修复后重训：D3 / D5 普通与混合对照');
merged(s,4,'H','本轮为新的模拟观察批次。所有主策略共用同一份Ridge32基线；D3和D5独立训练，未跨周期融合。');
s.getRange('A6:B7').values=[['信号日',date(m.as_of)],['每边费用',0.001]];
s.getRange('D6:E7').values=[['计划收盘入场',date(m.entry_date)],['每边模拟滑点',0.00075]];
s.getRange('G6:H7').values=[['可评分股票数',m.scored_count],['双边总成本',null]];
s.getRange('H7').formulas=[['=2*(B7+E7)']];s.getRange('B6').setNumberFormat('yyyy-mm-dd');s.getRange('E6').setNumberFormat('yyyy-mm-dd');s.getRange('B7').setNumberFormat(pct);s.getRange('E7').setNumberFormat('0.000%');s.getRange('H7').setNumberFormat(pct);
for(const [h,top] of [['3',9],['5',15]]){
 header(s,top,'H',['周期','策略','完整入选数','等量对照数','科技归类数','科技占比','到期观察日','用途']);
 const v=views[h],rows=x.unions[h],last=rows.length+8,detail=m.horizons[h];
 for(const [i,key] of ['ordinary','hybrid_no_flow','hybrid_flow'].entries()){
  const n=top+i+1,col=['E','F','G'][i],names=['普通Ridge32','混合：无资金流','混合：有资金流'];
  s.getRange(`A${n}:H${n}`).values=[[`D+${h}`,names[i],null,detail.equal_count_n,null,null,date(detail.target_date),'主策略模拟']];
  s.getRange(`C${n}`).formulas=[[`=COUNTIFS('${v.name}'!${col}9:${col}${last},">0")`]];
  s.getRange(`E${n}:F${n}`).formulas=[[`=COUNTIFS('${v.name}'!${col}9:${col}${last},">0",'${v.name}'!M9:M${last},"是")`,`=E${n}/C${n}`]];
  s.getRange(`G${n}`).setNumberFormat('yyyy-mm-dd');s.getRange(`F${n}`).setNumberFormat(pct);
  rowStats.push({row:n,h,key});
 }
 const n=top+4;
 s.getRange(`A${n}:H${n}`).values=[[`D+${h}`,'混合：分数Gate',null,'不参与',null,null,date(detail.target_date),'仅诊断']];
 s.getRange(`C${n}`).formulas=[[`=COUNTIFS('Gate诊断'!A9:A${x.diagnostic.length+8},A${n})`]];
 s.getRange(`E${n}:F${n}`).formulas=[[`=COUNTIFS('Gate诊断'!A9:A${x.diagnostic.length+8},A${n},'Gate诊断'!I9:I${x.diagnostic.length+8},"是")`,`=E${n}/C${n}`]];
 s.getRange(`G${n}`).setNumberFormat('yyyy-mm-dd');s.getRange(`F${n}`).setNumberFormat(pct);
}
merged(s,21,'H',`等量评审：D3各主策略前${m.horizons['3'].equal_count_n}只，D5前${m.horizons['5'].equal_count_n}只。数量事前固定，完整名单保留。`);
merged(s,22,'H','科技映射合计最多10只、每行业最多3只、总数最多50只；约束后不补位，故完整名单可能少于50只。');
header(s,24,'H',['周期','模型','输入因子数','训练起点','最后训练样本','训练行数','训练日数','目标字段']);
const trainingRows=[];
for(const h of ['3','5']){const z=m.horizons[h];trainingRows.push(
 [`D+${h}`,'共享Ridge',32,date(z.original_train_start),date(z.original_train_end),z.original_train_rows,'长期全量',z.label_col],
 [`D+${h}`,'近期无资金流',32,date(z.recent_train_start),date(z.recent_train_end),z.shared_recent_train_rows,z.shared_recent_dates,z.label_col],
 [`D+${h}`,'近期有资金流',37,date(z.recent_train_start),date(z.recent_train_end),z.shared_recent_train_rows,z.shared_recent_dates,z.label_col]);}
s.getRange('A25:H30').values=trainingRows;s.getRange('D25:E30').setNumberFormat('yyyy-mm-dd');s.getRange('F25:F30').setNumberFormat('#,##0');
merged(s,32,'H','标签定义：次日收盘至第h个交易日收盘收益，按交易所日历定位。持有窗口缺价的样本不训练。');
merged(s,33,'H','普通预测分是模型估计的原始远期收益；RRF是排名融合分，不是收益率。当前尚未入场，没有本轮回测收益。');
merged(s,34,'H','模型已保存供审计，未晋升生产注册模型。分数Gate只用于诊断，主策略按全量候选RRF排序。');
merged(s,35,'H',`陈旧数据排除：${m.excluded_symbols.join('、')}；其行情和资金流仍停在9月14日。其余299只均截至10月14日。`);
merged(s,36,'H','全池已核验完整前复权历史并重建特征与标签。旧版10月14日名单已撤回，本表为修复后的新批次。');
merged(s,37,'H','模拟费用与滑点为统一假设，后续收益按双边总成本35bp核对；不据本次训练认定准确率或跑赢指数。');
merged(s,38,'H','因子、模型参数和RRF保持既定口径。近10日涨跌和20日回撤可查；持续下跌过滤尚未纳入策略。');
widths(s,[10,24,16,20,20,16,18,21]);

for(const h of ['3','5']){
 const v=views[h],rows=x.unions[h],end=rows.length+8;
 setup(v,'O',end);merged(v,2,'O',`D+${h} 选股对照：普通、无资金流混合、有资金流混合`);
 v.getRange('A4:L5').values=[['信号日',date(m.as_of),null,'收盘入场',date(m.entry_date),null,'目标观察日',date(m.horizons[h].target_date),null,'等量评审N',m.horizons[h].equal_count_n,null],['普通入选',x.main_counts[h].ordinary,null,'无流入选',x.main_counts[h].hybrid_no_flow,null,'有流入选',x.main_counts[h].hybrid_flow,null,'最后成熟样本',date(m.horizons[h].original_train_end),null]];
 for(const cell of ['B4','E4','H4','K5'])v.getRange(cell).setNumberFormat('yyyy-mm-dd');
 merged(v,6,'O',`空白名次=未选中。按有流混合名次排列。等量评审各主组前${m.horizons[h].equal_count_n}只，历史涨跌数据仅作观察。`);
 header(v,8,'O',['股票代码','名称','行业','14日收盘','普通名次','无流混合名次','有流混合名次','普通预测分','基线全池排名','近期有流排名','有流RRF分','等量纳入策略','科技归类','近10日涨跌','近20日回撤']);
 v.getRange(`A9:O${end}`).values=rows.map(r=>[r.symbol,r.name,r.industry_name,r.close,r.ordinary_rank,r.no_flow_rank,r.flow_rank,r.base_score,r.base_rank,r.recent_rank,null,r.equal_membership,r.is_tech?'是':'否',r.return_10d,r.drawdown_20d]);
 v.getRange(`K9:K${end}`).formulas=rows.map((r,i)=>[`=1/(60+I${i+9})+1/(60+J${i+9})`]);
 v.getRange(`A9:A${end}`).setNumberFormat('@');v.getRange(`D9:D${end}`).setNumberFormat('0.00');v.getRange(`H9:H${end}`).setNumberFormat(pct);v.getRange(`K9:K${end}`).setNumberFormat('0.000000');
 rows.forEach((r,i)=>{if(r.flow_rank!==null&&r.flow_rank<=m.horizons[h].equal_count_n)v.getRange(`A${i+9}:M${i+9}`).format.fill='#EAF1F7';});
 v.getRange(`N9:O${end}`).setNumberFormat(pct);
 widths(v,[12,17,24,14,13,16,16,15,17,17,16,22,13,17,17]);v.freezePanes.freezeRows(8);
}

setup(g,'I',x.diagnostic.length+8);merged(g,2,'I','分数Gate诊断组：单独记录，不作为默认选股');
merged(g,4,'I','该组筛选A_MAIN/B_SHORT_BOOST再按RRF排序，与主策略有流组共用预测。仅作诊断，不决定主组等量数量。');
merged(g,5,'I','本轮无额外事件/风险因子；科技上限10只与主策略相同。未入场，未产生到期收益。');
header(g,8,'I',['周期','组内名次','股票代码','名称','行业','14日收盘','分数Gate层级','RRF分','科技归类']);
g.getRange(`A9:I${x.diagnostic.length+8}`).values=x.diagnostic.map(r=>[`D+${r.horizon}`,r.recommendation_rank,r.symbol,r.name,r.industry_name,r.close,r.gate_tier,r.fusion_score_col,x.tech_industries.includes(r.industry_name)?'是':'否']);
g.getRange(`C9:C${x.diagnostic.length+8}`).setNumberFormat('@');g.getRange(`F9:F${x.diagnostic.length+8}`).setNumberFormat('0.00');g.getRange(`H9:H${x.diagnostic.length+8}`).setNumberFormat('0.000000');widths(g,[10,13,13,18,25,15,24,17,15]);g.freezePanes.freezeRows(8);

setup(f,'G',58);merged(f,2,'G','实际训练因子与资金流使用证据');
merged(f,4,'G','D3/D5均独立训练：长期Ridge32，近期无流32列、有流37列；五个资金流因子各有120个完整训练日。');
merged(f,5,'G','分裂次数仅说明因子进入了树模型，不证明增量收益。近期有流/无流的行集合与训练参数一致。');
header(f,7,'G',['资金流因子','D3训练日数','D3当日覆盖','D3分裂次数','D5训练日数','D5当日覆盖','D5分裂次数']);
const labels={'cs_main_flow_rank_1d':'当日主力净流入排名','cs_main_flow_rank_5d':'近5日主力净流入排名','cs_small_flow_rank_1d':'当日小单净流入排名','cs_super_flow_rank_1d':'当日超大单净流入排名','cs_flow_reversal_5d':'主力流入短期变化'};
f.getRange('A8:G12').values=m.flow_features.map(c=>[labels[c],m.horizons['3'].flow_train_nonnull_dates[c],m.scored_count,m.horizons['3'].flow_importances[c],m.horizons['5'].flow_train_nonnull_dates[c],m.scored_count,m.horizons['5'].flow_importances[c]]);
merged(f,14,'G','资金流按日期对应的流通市值归一化；补齐缺失市值日期，未将未来股本倒填到过去，未补填本轮不用的PE/PB。');
merged(f,15,'G','历史资金流是事后补齐数据，供应商修订及历史可得性未完成独立审计；本轮作为事前模拟观察起点。');
header(f,18,'G',['因子字段','Ridge使用','近期无流','近期有流','类型','D3分裂次数','D5分裂次数']);
const fields=[...m.original_features,...m.flow_features];f.getRange(`A19:G${fields.length+18}`).values=fields.map(c=>{const flow=m.flow_features.includes(c);return[c,flow?'否':'是',flow?'否':'是','是',flow?'资金流':'原版因子',x.importance['3'][c],x.importance['5'][c]];});widths(f,[33,18,18,18,17,17,17]);f.freezePanes.freezeRows(18);

setup(p,'K',x.coverage.length+8);merged(p,2,'K','行情与数据覆盖：300只股票逐项核对');
merged(p,4,'K','行情源：AKShare新浪前复权日线，https://akshare.akfamily.xyz/data/stock/stock.html；资金流源：东方财富H5。');
merged(p,5,'K','资金流API：https://emdatah5.eastmoney.com/dc/ZJLX/getDBHistoryData；原始量为元，流通市值为亿元。');
merged(p,6,'K','新增流通市值按当日收盘×当日流通股本/1亿计算。行情、资金流及市值在信号日截断；预测记录已冻结。');
header(p,8,'K',['股票代码','名称','行情截止','资金流截止','市值截止','最新收盘价','主力净流入(元)','小单净流入(元)','超大单净流入(元)','流通市值(亿)','覆盖状态']);
p.getRange(`A9:K${x.coverage.length+8}`).values=x.coverage.map(r=>[r.symbol,r.name,date(r.price_date),date(r.flow_date),date(r.cap_date),r.close,r.main_net,r.small_net,r.super_net,r.float_mcap_yi,r.status]);
p.getRange(`A9:A${x.coverage.length+8}`).setNumberFormat('@');p.getRange(`C9:E${x.coverage.length+8}`).setNumberFormat('yyyy-mm-dd');p.getRange(`F9:F${x.coverage.length+8}`).setNumberFormat('0.00');p.getRange(`G9:I${x.coverage.length+8}`).setNumberFormat('#,##0');p.getRange(`J9:J${x.coverage.length+8}`).setNumberFormat('#,##0.00');widths(p,[13,18,18,18,18,15,24,24,26,21,23]);p.freezePanes.freezeRows(8);

w.recalculate();
for(const {row,h,key} of rowStats){if(s.getRange(`C${row}`).values[0][0]!==x.main_counts[h][key])throw Error('Selection count mismatch');if(s.getRange(`E${row}`).values[0][0]>10)throw Error('Tech cap violated');}
for(const h of ['3','5'])x.unions[h].forEach((r,i)=>{if(Math.abs(views[h].getRange(`K${i+9}`).values[0][0]-r.fusion_score)>1e-12)throw Error('RRF formula mismatch');});
const old=s.getRange('B7').values[0][0];s.getRange('B7').values=[[old+0.001]];w.recalculate();if(Math.abs(s.getRange('H7').values[0][0]-0.0055)>1e-12)throw Error('Cost recalculation failed');s.getRange('B7').values=[[old]];w.recalculate();
console.log((await w.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!',options:{useRegex:true,maxResults:10},maxChars:600})).ndjson);
// Verify editable cost and one RRF formula, then restore the frozen inputs.
s.getRange('B7').values=[[0.0015]];w.recalculate();
if(Math.abs(s.getRange('H7').values[0][0]-0.0045)>1e-12)throw new Error('Cost recalculation failed');
s.getRange('B7').values=[[0.001]];
const checkView=views['3'],oldBaseRank=checkView.getRange('I9').values[0][0],recentRank=checkView.getRange('J9').values[0][0];
checkView.getRange('I9').values=[[oldBaseRank+1]];w.recalculate();
if(Math.abs(checkView.getRange('K9').values[0][0]-(1/(61+oldBaseRank)+1/(60+recentRank)))>1e-12)throw new Error('RRF recalculation failed');
checkView.getRange('I9').values=[[oldBaseRank]];w.recalculate();
await(await SpreadsheetFile.exportXlsx(w)).save(out);console.log(out);
