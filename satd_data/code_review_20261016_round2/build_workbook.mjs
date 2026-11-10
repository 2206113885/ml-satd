import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';

const base=path.dirname(fileURLToPath(import.meta.url));
const payload=JSON.parse(await fs.readFile(path.join(base,'review_summary.json'),'utf8'));
const wb=Workbook.create();
const overview=wb.worksheets.add('问题总表');
const sheet=wb.worksheets.add('修改清单');
const coverage=wb.worksheets.add('覆盖记录');
const clean=v=>typeof v==='string'&&v.startsWith('=')?"'"+v:v;

function init(s,title,note,headers,rows,widths,tableName){
  s.showGridLines=false;
  const last=String.fromCharCode(64+headers.length);
  const end=rows.length+6;
  s.getRange(`A1:${last}${end}`).format.font={name:'Arial',size:10,color:'#243447'};
  s.getRange('A2').values=[[title]];
  s.getRange('A2').format.font={name:'Arial',size:14,bold:true,color:'#243E59'};
  s.getRange(`A3:${last}3`).format.borders={bottom:{style:'thin',color:'#AFC1D2'}};
  s.getRange('A4').values=[[note]];
  s.getRange('A4').format.font={name:'Arial',size:10,color:'#526273'};
  s.getRange(`A6:${last}6`).values=[headers];
  if(rows.length)s.getRange(`A7:${last}${end}`).values=rows.map(row=>row.map(clean));
  const t=s.tables.add(`A6:${last}${end}`,true,tableName);
  t.showFilterButton=true;
  s.getRange(`A6:${last}${end}`).format.wrapText=true;
  s.getRange(`A6:${last}${end}`).format.verticalAlignment='top';
  s.getRange(`A6:${last}6`).format={fill:'#243E59',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,horizontalAlignment:'center',verticalAlignment:'center',rowHeight:28};
  widths.forEach((w,i)=>s.getRange(`${String.fromCharCode(65+i)}1:${String.fromCharCode(65+i)}${end}`).format.columnWidthPx=w);
  s.freezePanes.freezeRows(6);
  s.freezePanes.freezeColumns(s===sheet?1:1);
  rows.forEach((row,idx)=>{
    const lineCount=Math.max(...row.map((v,col)=>{
      const str=String(v??'');
      const visual=[...str].reduce((a,c)=>a+(/[\u2e80-\uffff]/.test(c)?1.9:1),0);
      return Math.ceil(visual/Math.max(6,(widths[col]-16)/7));
    }));
    s.getRange(`A${idx+7}:${last}${idx+7}`).format.rowHeight=Math.max(40,Math.min(330,lineCount*15+12));
  });
  return end;
}

const rows=payload.locations.map(r=>[
  r.issue_id,r.severity,r.status_label,r.file,r.line_display,r.title,
  r.recommendation,r.impact,r.manual_status,''
]);
init(overview,'实验代码审计问题总表',
  `${payload.code_files_reviewed}个脚本 · ${payload.finding_count}类事项：25类确认、12类风险、3类待核实、4类设计选择。逐文件位置及处理状态见修改清单。`,
  ['问题编号','优先级','判断','文件数','问题','修改建议','影响与边界'],
  payload.groups.map(f=>[f.issue_id,f.severity,f.status_label,f.affected_files,f.title,f.recommendation,f.impact]),
  [90,65,100,70,300,550,560],'CodeReviewIssues');
overview.tabColor='#243E59';
const end=init(sheet,'实验代码修改清单',
  `${payload.code_files_reviewed}个脚本 · ${payload.finding_count}类事项 · ${rows.length}个文件事项。判断指当前代码；历史分数影响另核实。黄色列供手动填写。`,
  ['问题编号','优先级','判断','文件（相对 satd_data）','行号','问题','修改建议','影响与边界','处理状态','修改备注'],
  rows,[90,65,100,345,105,235,500,470,125,300],'CodeReviewActions');
sheet.tabColor='#243E59';
sheet.getRange(`I7:J${end}`).format.fill='#FFF3CE';
sheet.getRange(`I7:I${end}`).dataValidation={rule:{type:'list',values:['待处理','修改中','已修改','待核实','待决定','不修改（已说明）']}};
sheet.getRange(`I7:I${end}`).conditionalFormats.add('containsText',{text:'已修改',format:{fill:'#E4F1E8',font:{color:'#245A38'}}});
sheet.getRange(`B7:B${end}`).conditionalFormats.add('cellIs',{operator:'equal',formula:'"P1"',format:{fill:'#FBE9E7',font:{bold:true,color:'#A53226'}}});
const coverRows=payload.coverage.map(r=>[r.file,r.role,r.reviewed_areas,r.issue_ids.join(', ')||'未发现额外问题',r.sha256]);
init(coverage,'审查覆盖记录','原始脚本114个，先导实验10个。未发现问题不代表已验证全部历史运行；审计工具自身不计入实验代码。',
  ['文件（相对 satd_data）','用途','已审范围','关联事项','审查时 SHA-256'],
  coverRows,[380,275,530,360,460],'CodeReviewCoverage');

wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'修改清单!A6:J9',include:'values,formulas',tableMaxRows:4,tableMaxCols:10,tableMaxCellChars:75,maxChars:5000})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},summary:'final formula error scan',maxChars:1500})).ndjson);
for(const [name,range,out] of [['问题总表','A1:G11','overview_preview.png'],['修改清单','A1:H10','actions_preview.png'],['修改清单','I6:J12','manual_status_preview.png'],['覆盖记录','A1:E11','coverage_preview.png']]){
  const blob=await wb.render({sheetName:name,range,scale:1,format:'png'});
  await fs.writeFile(path.join(base,out),new Uint8Array(await blob.arrayBuffer()));
}
const outDir=path.join(base,'outputs','01a1439c-689e-72b1-82c0-2c50695a0417');
await fs.mkdir(outDir,{recursive:true});
const outPath=path.join(outDir,'实验代码审计与修改清单.xlsx');
await (await SpreadsheetFile.exportXlsx(wb)).save(outPath);
console.log(JSON.stringify({output:outPath,action_rows:rows.length,coverage_rows:coverRows.length}));
