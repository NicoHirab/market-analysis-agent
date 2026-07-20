from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse

from market_agent.api.schemas import AnalysisResource, AnalysisSummary, AnalyzeRequest
from market_agent.api.service import AnalysisService

router = APIRouter(prefix="/api/v1")


def _service(request: Request) -> AnalysisService:
    return request.app.state.service


@router.post("/analyses", response_model=AnalysisResource, status_code=202)
async def create_analysis(
    body: AnalyzeRequest, request: Request, response: Response, wait: bool = False
):
    svc = _service(request)
    analyses = None if body.analyses == "auto" else list(body.analyses)
    job = await svc.start(
        body.product, language=body.language, analyses=analyses, platforms=body.platforms
    )
    if wait:
        job = await svc.wait(job.id)
        response.status_code = 200
    return AnalysisResource.from_job(job)


@router.get("/analyses", response_model=list[AnalysisSummary])
async def list_analyses(request: Request):
    return [AnalysisSummary.from_job(j) for j in _service(request).registry.list_jobs()]


def _job_or_404(request: Request, analysis_id: str):
    job = _service(request).registry.get(analysis_id)
    if job is None:
        raise HTTPException(status_code=404, detail="analysis not found")
    return job


@router.get("/analyses/{analysis_id}", response_model=AnalysisResource)
async def get_analysis(analysis_id: str, request: Request):
    return AnalysisResource.from_job(_job_or_404(request, analysis_id))


@router.get("/analyses/{analysis_id}/report.md", response_class=PlainTextResponse)
async def get_report_markdown(analysis_id: str, request: Request):
    job = _job_or_404(request, analysis_id)
    markdown = (job.result or {}).get("report_markdown")
    if not markdown:
        raise HTTPException(status_code=404, detail="report not available")
    return PlainTextResponse(markdown, media_type="text/markdown; charset=utf-8")
