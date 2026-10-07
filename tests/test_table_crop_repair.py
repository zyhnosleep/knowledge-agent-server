import base64

import pytest

from app.services import table_crop_repair as repair
from app.services.canonical_models import CanonicalBlock, CanonicalCell, CanonicalDocument, CanonicalTable, SourceSpan
from app.services.canonical_quality import CanonicalQualityGate

PNG=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jB1sAAAAASUVORK5CYII=')


def fixture(tmp_path, monkeypatch):
    root = tmp_path / 'cache'
    base = root / 'mineru' / 'run'
    base.mkdir(parents=True)
    (base/'table.png').write_bytes(PNG)
    monkeypatch.setattr(repair.settings, 'cache_dir', root)
    monkeypatch.setattr(repair.settings, 'mineru_output_dir', root/'mineru')
    headers = ['Method', 'HotpotQA (EM)']
    rows = [['CoT CoT-SC', '29.4 33.4']]
    table = CanonicalTable(table_id='t1', caption='Table 1: Results', headers=headers, rows=rows,
        cells=[CanonicalCell(text=v,row_index=r,column_index=c,is_header=r==0)
            for r,row in enumerate([headers,*rows]) for c,v in enumerate(row)],
        source_spans=[SourceSpan(page_index=4,page_label='5',bbox=(173,101,449,244),source_block_id='mineru-table-1')],
        metadata={'image_path':'table.png'})
    doc=CanonicalDocument(document_id='d1',parse_version='v1',parser_source='mineru',tables=[table],
        blocks=[CanonicalBlock(block_id='b1',block_type='table',parser_source='mineru',text='OLD MERGED SCORES',table_id='t1',reading_order=0,
                    source_spans=table.source_spans),
                CanonicalBlock(block_id='b2',block_type='narrative',parser_source='mineru',text='Unchanged narrative',reading_order=1)],
        parser_metadata={'source_parser_metadata':{'document_intelligence':{'output_dir':str(base)}}})
    issues=CanonicalQualityGate().evaluate(doc).issues
    return doc,[i for i in issues if i.repairable],base


class Reader:
    def __init__(self, markdown='| Method | HotpotQA (EM) |\n| --- | --- |\n| CoT | 29.4 |\n| CoT-SC | 33.4 |'):
        self.markdown=markdown
        self.inputs=[]

    def generate_structured_with_images(self,schema,**kwargs):
        self.inputs.append(kwargs)
        return schema(markdown=self.markdown)


def test_crop_binds_actual_pixels_to_trusted_locator_and_updates_source_blocks(tmp_path,monkeypatch):
    doc,issues,base=fixture(tmp_path,monkeypatch)
    before=doc.model_dump()
    client=Reader()
    fixed=repair.repair_table_crops(doc,issues,client=client)
    assert fixed is not None and fixed.quality.accepted
    assert fixed.tables[0].rows == [['CoT','29.4'],['CoT-SC','33.4']]
    assert fixed.tables[0].source_spans == doc.tables[0].source_spans
    proof=fixed.tables[0].metadata['repair_proof']
    assert proof['validated_mapping']['original_table_id']=='t1'
    assert proof['page_index']==4
    assert 'OLD MERGED SCORES' not in fixed.blocks[0].text and '33.4' in fixed.blocks[0].text
    assert fixed.blocks[1].text=='Unchanged narrative'
    assert doc.model_dump()==before
    assert client.inputs[0]['images']==[(base/'table.png').read_bytes()]


@pytest.mark.parametrize('bad_path',['../outside.png','/etc/passwd','https://example.com/img.png'])
def test_unsafe_crop_never_reaches_model(tmp_path,monkeypatch,bad_path):
    doc,issues,_=fixture(tmp_path,monkeypatch)
    doc.tables[0].metadata['image_path']=bad_path
    client=Reader()
    assert repair.repair_table_crops(doc,issues,client=client) is None
    assert client.inputs==[]


def test_nonimage_crop_and_merged_model_scores_remain_blocked(tmp_path,monkeypatch):
    doc,issues,base=fixture(tmp_path,monkeypatch)
    (base/'table.png').write_bytes(b'not an image')
    client=Reader()
    assert repair.repair_table_crops(doc,issues,client=client) is None
    assert client.inputs==[]
    (base/'table.png').write_bytes(PNG)
    assert repair.repair_table_crops(doc,issues,client=Reader('| Method | HotpotQA (EM) |\n| --- | --- |\n| mixed | 29.4 33.4 |')) is None


def test_missing_source_locator_cannot_be_invented_from_caption(tmp_path,monkeypatch):
    doc,issues,_=fixture(tmp_path,monkeypatch)
    doc.tables[0].source_spans=[SourceSpan(page_index=4,page_label='5')]
    assert repair.repair_table_crops(doc,issues,client=Reader()) is None


def test_crop_outside_declared_output_root_is_rejected(tmp_path,monkeypatch):
    doc,issues,base=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(repair.settings,'mineru_output_dir',tmp_path/'other')
    client=Reader()
    assert repair.repair_table_crops(doc,issues,client=client) is None
    assert client.inputs==[]


def test_pipeline_uses_crop_inventory_before_unbound_whole_page_repair(tmp_path,monkeypatch):
    from app.services import canonical_adapters, parser
    from app.services.pipeline import IngestionPipeline
    doc,issues,base=fixture(tmp_path,monkeypatch)
    doc.metadata.update(expected_page_count=5,parsed_page_indices=[0,1,2,3,4],text_layer_pages=['']*5)
    monkeypatch.setattr(parser.settings,'document_intelligence_enabled',True)
    reader=Reader()
    monkeypatch.setattr(repair.OllamaClient,'generate_structured_with_images',lambda _self,*args,**kwargs:reader.generate_structured_with_images(*args,**kwargs))
    def whole_page(*args,**kwargs):
        raise AssertionError('crop-backed repair should not need unbound page replacement')
    monkeypatch.setattr(canonical_adapters,'run_document_intelligence',whole_page)
    result=IngestionPipeline(None)._repair_canonical_phase(doc,tmp_path/'paper.pdf')
    assert result.quality.accepted
    assert result.tables[0].rows==[['CoT','29.4'],['CoT-SC','33.4']]
    assert result.tables[0].metadata['repair_proof_validated'] is True
    assert 'OLD MERGED SCORES' not in result.blocks[0].text


def test_symlink_crop_is_rejected_before_model(tmp_path,monkeypatch):
    doc,issues,base=fixture(tmp_path,monkeypatch)
    link=base/'linked.png'
    try:
        link.symlink_to(base/'table.png')
    except OSError:
        pytest.skip('Windows symlink permission unavailable; must run on server')
    doc.tables[0].metadata['image_path']='linked.png'
    client=Reader()
    assert repair.repair_table_crops(doc,issues,client=client) is None
    assert client.inputs==[]
