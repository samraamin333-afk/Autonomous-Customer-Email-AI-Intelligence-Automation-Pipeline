"""
Daily Management Report Generator.
Aggregates and formats business metrics:
- Total emails processed
- Category distribution (counts & percentages)
- Urgent cases requiring management escalation
- Pending issues & SLA tracking
- Common problem clusters
Outputs both structured Markdown and standalone HTML executive dashboards.
"""

from typing import List, Dict, Any
from datetime import datetime
from collections import Counter
import json
import os


class DailyReportGenerator:
    """Generates executive summaries and operational reports from processed email data."""

    def __init__(self, processed_records: List[Dict[str, Any]]):
        self.records = processed_records

    def generate_metrics(self) -> Dict[str, Any]:
        """Calculates aggregated metrics across all processed emails."""
        total_emails = len(self.records)
        if total_emails == 0:
            return {"total_emails": 0}

        # Category distribution
        categories = [r.get("category", "Unknown") for r in self.records]
        cat_counts = Counter(categories)
        cat_dist = {cat: {"count": count, "percentage": round((count / total_emails) * 100, 1)}
                    for cat, count in cat_counts.most_common()}

        # Urgency distribution
        urgencies = [r.get("urgency_level", "LOW") for r in self.records]
        urgency_counts = Counter(urgencies)
        urgency_dist = {u: urgency_counts.get(u, 0) for u in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]}

        # Urgent / Escalation cases (Critical or High)
        urgent_cases = [
            r for r in self.records
            if r.get("urgency_level") in ["CRITICAL", "HIGH"]
        ]

        # Common problem patterns
        problems = [r.get("summary", {}).get("customer_problem", "") for r in self.records]

        # Preprocessing efficiency metrics
        reductions = [r.get("preprocessing", {}).get("character_reduction_pct", 0) for r in self.records]
        avg_reduction = round(sum(reductions) / len(reductions), 1) if reductions else 0.0

        return {
            "report_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "total_emails": total_emails,
            "category_distribution": cat_dist,
            "urgency_distribution": urgency_dist,
            "critical_count": urgency_dist.get("CRITICAL", 0),
            "high_count": urgency_dist.get("HIGH", 0),
            "urgent_cases": urgent_cases,
            "common_problems": problems,
            "avg_preprocessing_reduction_pct": avg_reduction,
        }

    def generate_markdown(self) -> str:
        """Renders report in GitHub Markdown format."""
        m = self.generate_metrics()
        if m.get("total_emails", 0) == 0:
            return "# Daily Email Management Report\n\nNo emails processed today."

        md = []
        md.append(f"# 📊 Daily Customer Email Intelligence & Operations Report")
        md.append(f"**Generated:** {m['report_date']} | **Total Emails Processed:** {m['total_emails']}\n")

        md.append("## 📈 Executive Summary & Key Performance Indicators")
        md.append(f"- **Total Volume Ingested:** {m['total_emails']} messages")
        md.append(f"- **Urgent Escalations (Critical/High):** {m['critical_count'] + m['high_count']} cases ({round(((m['critical_count'] + m['high_count'])/m['total_emails'])*100, 1)}% of total)")
        md.append(f"- **Avg Noise Reduction (HTML/Signatures):** {m['avg_preprocessing_reduction_pct']}% payload reduction\n")

        md.append("## 🗂️ Category Distribution")
        md.append("| Category | Ticket Count | Share (%) |")
        md.append("| :--- | :---: | :---: |")
        for cat, data in m["category_distribution"].items():
            md.append(f"| {cat} | {data['count']} | {data['percentage']}% |")
        md.append("")

        md.append("## 🚨 Urgent Cases Requiring Priority Action (SLA < 4 hrs)")
        if not m["urgent_cases"]:
            md.append("*No urgent or critical tickets detected today.*")
        else:
            md.append("| Email ID | Urgency | Sender | Category | Customer Problem | SLA Target |")
            md.append("| :--- | :---: | :--- | :--- | :--- | :---: |")
            for u in m["urgent_cases"]:
                sla = u.get("sla_target_hours", 24)
                prob = u.get("summary", {}).get("customer_problem", "N/A")[:55] + "..."
                md.append(f"| `{u.get('id')}` | **{u.get('urgency_level')}** | `{u.get('sender')}` | {u.get('category')} | {prob} | {sla}h |")
        md.append("")

        md.append("## 🔍 Common Problems & Operational Themes")
        for i, r in enumerate(self.records, 1):
            sum_data = r.get("summary", {})
            md.append(f"{i}. **[{r.get('category')}]** {sum_data.get('customer_problem')}")
            md.append(f"   - *Action Requested:* {sum_data.get('requested_action')}")
            if sum_data.get("important_details_summary") != "None specified":
                md.append(f"   - *Key Details:* {sum_data.get('important_details_summary')}")
        md.append("")

        md.append("---")
        md.append("*Report generated automatically by Customer Email AI Automation System.*")
        return "\n".join(md)

    def generate_html(self, output_path: str = "data/daily_management_report.html") -> str:
        """Renders an interactive executive HTML dashboard."""
        m = self.generate_metrics()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        rows = ""
        for u in m.get("urgent_cases", []):
            badge_color = "#dc2626" if u.get("urgency_level") == "CRITICAL" else "#ea580c"
            rows += f"""
            <tr>
                <td><code>{u.get('id')}</code></td>
                <td><span style="background:{badge_color}; color:#fff; padding:2px 8px; border-radius:12px; font-weight:600; font-size:12px;">{u.get('urgency_level')}</span></td>
                <td>{u.get('sender')}</td>
                <td>{u.get('category')}</td>
                <td>{u.get('summary', {}).get('customer_problem')}</td>
                <td><strong>{u.get('sla_target_hours')}h</strong></td>
            </tr>
            """

        cat_rows = ""
        for cat, data in m.get("category_distribution", {}).items():
            cat_rows += f"""
            <tr>
                <td>{cat}</td>
                <td style="text-align:center;"><strong>{data['count']}</strong></td>
                <td style="text-align:right;">
                    <div style="background:#e2e8f0; border-radius:4px; height:12px; width:100px; display:inline-block; margin-right:8px;">
                        <div style="background:#3b82f6; height:12px; border-radius:4px; width:{data['percentage']}%;"></div>
                    </div>
                    {data['percentage']}%
                </td>
            </tr>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Daily Management Report - Email Intelligence</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 24px; background: #f8fafc; color: #1e293b; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #e2e8f0; padding-bottom: 16px; margin-bottom: 24px; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }}
        .kpi-card {{ background: #ffffff; border-radius: 8px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border: 1px solid #e2e8f0; }}
        .kpi-title {{ font-size: 13px; color: #64748b; font-weight: 600; text-transform: uppercase; margin-bottom: 6px; }}
        .kpi-value {{ font-size: 28px; font-weight: 700; color: #0f172a; }}
        .panel {{ background: #ffffff; border-radius: 8px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border: 1px solid #e2e8f0; margin-bottom: 24px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #e2e8f0; font-size: 14px; }}
        th {{ background: #f1f5f9; font-weight: 600; color: #475569; }}
        code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1 style="margin:0 0 6px 0; font-size:24px;">📧 Customer Email Operations Dashboard</h1>
            <span style="color:#64748b; font-size:14px;">Report Run: {m['report_date']}</span>
        </div>
        <div>
            <span style="background:#22c55e; color:#fff; padding:6px 14px; border-radius:16px; font-weight:600; font-size:13px;">Pipeline Active</span>
        </div>
    </div>

    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">Total Ingested</div>
            <div class="kpi-value">{m['total_emails']}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Critical / High Urgency</div>
            <div class="kpi-value" style="color:#dc2626;">{m['critical_count'] + m['high_count']}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Avg Noise Reduction</div>
            <div class="kpi-value" style="color:#0284c7;">{m['avg_preprocessing_reduction_pct']}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Auto-Drafted Responses</div>
            <div class="kpi-value" style="color:#16a34a;">{m['total_emails']}</div>
        </div>
    </div>

    <div class="panel">
        <h2 style="margin-top:0; font-size:18px;">🚨 Priority Escalation Queue (SLA &lt; 4 Hours)</h2>
        <table>
            <thead>
                <tr>
                    <th>Email ID</th>
                    <th>Urgency</th>
                    <th>Sender</th>
                    <th>Category</th>
                    <th>Problem Summary</th>
                    <th>Target SLA</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>

    <div class="panel">
        <h2 style="margin-top:0; font-size:18px;">📊 Volume by Category</h2>
        <table>
            <thead>
                <tr>
                    <th>Category</th>
                    <th style="text-align:center;">Volume</th>
                    <th style="text-align:right;">Proportion</th>
                </tr>
            </thead>
            <tbody>
                {cat_rows}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return output_path
