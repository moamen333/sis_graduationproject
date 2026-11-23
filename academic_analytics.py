
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


class AcademicAnalytics:
    """Academic performance analytics for Uni Assist."""

    def __init__(self, data_dir=None):
        self.data_dir = (
            Path(data_dir)
            if data_dir is not None
            else Path(__file__).resolve().parent / "uni_assist_dataset"
        )

        self.students = pd.read_csv(self.data_dir / "students.csv")
        self.courses = pd.read_csv(self.data_dir / "courses.csv")
        self.grades = pd.read_csv(self.data_dir / "grades.csv")
        self.activity = pd.read_csv(
            self.data_dir / "student_activity.csv"
        )

    def _student_grades(self, student_id):
        grades = self.grades[
            self.grades["student_id"].astype(str) == str(student_id)
        ].copy()

        grades["numeric_grade"] = pd.to_numeric(
            grades["numeric_grade"], errors="coerce"
        )
        return grades.dropna(subset=["numeric_grade"])

    def _show_one_chart(self, fig):
        """Display one plot and wait until its window is closed."""
        fig.tight_layout()
        plt.show(block=True)
        plt.close(fig)

    def create_semester_chart(self, student_id):
        trends = self.analyze_student(student_id)["semester_trends"]

        if not trends:
            print("No semester data available.")
            return False

        semesters = [item["semester"] for item in trends]
        averages = [item["average_grade"] for item in trends]

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(semesters, averages, marker="o", linewidth=2)
        ax.set_title(f"Semester Performance - {student_id}")
        ax.set_xlabel("Semester")
        ax.set_ylabel("Average Grade")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=20)

        for semester, average in zip(semesters, averages):
            ax.annotate(
                f"{average:.1f}",
                (semester, average),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
            )

        self._show_one_chart(fig)
        return True

    def create_course_performance_chart(self, student_id):
        grades = self._student_grades(student_id)

        if grades.empty:
            print("No course grades available.")
            return False

        results = (
            grades.groupby("course_id", as_index=False)["numeric_grade"]
            .mean()
            .merge(
                self.courses[["course_id", "course_name"]],
                on="course_id",
                how="left",
            )
            .sort_values("numeric_grade")
        )

        labels = results["course_name"].fillna(
            results["course_id"]
        ).astype(str)

        fig, ax = plt.subplots(
            figsize=(10, max(5, 0.38 * len(results)))
        )
        bars = ax.barh(labels, results["numeric_grade"])
        ax.set_title(f"Average Grade by Course - {student_id}")
        ax.set_xlabel("Average Grade")
        ax.set_xlim(0, 100)
        ax.grid(axis="x", alpha=0.25)
        ax.bar_label(bars, fmt="%.1f", padding=3)

        self._show_one_chart(fig)
        return True

    def create_pass_fail_chart(self, student_id):
        grades = self._student_grades(student_id)

        if grades.empty:
            print("No grade data available.")
            return False

        passed = int((grades["numeric_grade"] >= 60).sum())
        failed = int((grades["numeric_grade"] < 60).sum())

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(
            [passed, failed],
            labels=[
                f"Passing (60+): {passed}",
                f"Below passing: {failed}",
            ],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title(f"Pass / Fail Distribution - {student_id}")
        ax.axis("equal")

        self._show_one_chart(fig)
        return True

    def create_grade_distribution_chart(self, student_id):
        grades = self._student_grades(student_id)

        if grades.empty:
            print("No grade data available.")
            return False

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.hist(grades["numeric_grade"], bins=10, edgecolor="black")
        ax.axvline(
            60,
            linestyle="--",
            linewidth=2,
            label="Passing threshold (60)",
        )
        ax.set_title(f"Grade Distribution - {student_id}")
        ax.set_xlabel("Grade")
        ax.set_ylabel("Number of Records")
        ax.set_xlim(0, 100)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)

        self._show_one_chart(fig)
        return True

    def create_activity_charts(self, student_id):
        activity = self.activity[
            self.activity["student_id"].astype(str) == str(student_id)
        ].copy()

        if activity.empty:
            print("No activity data available.")
            return 0

        metrics = [
            ("attendance_pct", "Attendance", "Attendance (%)", 0, 100),
            ("study_hours_per_week", "Study Hours per Week", "Hours", 0, None),
            ("quiz_avg", "Quiz Average", "Average Score", 0, 100),
            (
                "assignment_completion_pct",
                "Assignment Completion",
                "Completion (%)",
                0,
                100,
            ),
        ]

        count = 0

        for column, title, ylabel, ymin, ymax in metrics:
            if column not in activity.columns:
                continue

            values = pd.to_numeric(
                activity[column], errors="coerce"
            ).dropna()

            if values.empty:
                continue

            average = float(values.mean())
            upper = ymax if ymax is not None else max(1, average * 1.25)

            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.bar([title], [average], width=0.5)
            ax.set_title(f"{title} - {student_id}")
            ax.set_ylabel(ylabel)
            ax.set_ylim(ymin, upper)
            ax.grid(axis="y", alpha=0.25)
            ax.bar_label(bars, fmt="%.2f", padding=5)

            self._show_one_chart(fig)
            count += 1

        return count

    def create_support_indicators_chart(self, student_id):
        result = self.analyze_student(student_id)
        activity = result["activity"]
        performance = result["performance"]

        indicators = [
            ("Average grade below 60",
             performance["average_grade"] < 60),
            ("Grades below passing",
             performance["failed_records"] > 0),
            ("Attendance below 75%",
             activity["attendance_pct"] is not None
             and activity["attendance_pct"] < 75),
            ("Assignment completion below 70%",
             activity["assignment_completion_pct"] is not None
             and activity["assignment_completion_pct"] < 70),
        ]

        labels = [label for label, _ in indicators]
        values = [int(triggered) for _, triggered in indicators]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(labels, values)
        ax.set_xlim(0, 1.5)
        ax.set_xticks([0, 1], ["Not triggered", "Triggered"])
        ax.set_title(f"Academic Support Indicators - {student_id}")
        ax.grid(axis="x", alpha=0.25)
        ax.bar_label(
            bars,
            labels=[
                "Triggered" if value else "Not triggered"
                for value in values
            ],
            padding=3,
        )
        fig.text(
            0.5, 0.01,
            "Prototype rules only; these are not validated predictions.",
            ha="center",
            fontsize=9,
        )

        self._show_one_chart(fig)
        return True

    def create_all_charts(self, student_id):
        """Display plots one at a time without saving image files."""
        self.analyze_student(student_id)
        count = 0

        methods = [
            self.create_semester_chart,
            self.create_course_performance_chart,
            self.create_pass_fail_chart,
            self.create_grade_distribution_chart,
        ]

        for method in methods:
            if method(student_id):
                count += 1

        count += self.create_activity_charts(student_id)

        if self.create_support_indicators_chart(student_id):
            count += 1

        print(f"\nFinished displaying {count} plots.")
        return count

    def create_dashboard(self, student_id):
        result = self.analyze_student(student_id)
        grades = self._student_grades(student_id)

        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(
        f"Academic Analytics Dashboard - {student_id}",
        fontsize=18,
        fontweight="bold"
        )

        # 1. Semester Performance
        ax = axes[0, 0]
        trends = result["semester_trends"]

        if trends:
            semesters = [x["semester"] for x in trends]
            averages = [x["average_grade"] for x in trends]

            ax.plot(semesters, averages, marker="o")
            ax.set_ylim(0, 100)
            ax.tick_params(axis="x", rotation=20)
            ax.set_title("Semester Performance")
            ax.set_ylabel("Average Grade")
            ax.grid(alpha=0.3)
        else:
            ax.set_title("No Semester Data")

         # 2. Course Performance
        ax = axes[0, 1]

        course_results = (
        grades.groupby("course_id")["numeric_grade"]
        .mean()
        .reset_index()
        .merge(
            self.courses[["course_id", "course_name"]],
            on="course_id",
            how="left"
        )
        )

        course_results = course_results.sort_values("numeric_grade")

        ax.barh(
        course_results["course_name"],
        course_results["numeric_grade"]
        )
        ax.set_xlim(0, 100)
        ax.set_title("Course Performance")
        ax.grid(axis="x", alpha=0.3)

        # 3. Pass / Fail
        ax = axes[1, 0]

        passed = (grades["numeric_grade"] >= 60).sum()
        failed = (grades["numeric_grade"] < 60).sum()

        ax.pie(
        [passed, failed],
        labels=["Passed", "Failed"],
        autopct="%1.1f%%",
        startangle=90
        )
        ax.set_title("Pass / Fail Distribution")

        # 4. Grade Distribution
        ax = axes[1, 1]

        ax.hist(
        grades["numeric_grade"],
        bins=10,
        edgecolor="black"
        )
        ax.axvline(60, linestyle="--", label="Passing Grade")
        ax.set_xlim(0, 100)
        ax.set_title("Grade Distribution")
        ax.set_xlabel("Grade")
        ax.set_ylabel("Frequency")
        ax.legend()

        # 5. Student Activity
        ax = axes[2, 0]

        activity = result["activity"]

        metrics = {
        "Attendance": activity["attendance_pct"],
        "Quiz Average": activity["quiz_avg"],
        "Assignments": activity["assignment_completion_pct"]
        }

        available = {
        key: value for key, value in metrics.items()
        if value is not None
        }

        if available:
            ax.bar(available.keys(), available.values())
            ax.set_ylim(0, 100)
            ax.set_title("Student Activity")
            ax.tick_params(axis="x", rotation=15)
            ax.grid(axis="y", alpha=0.3)
        else:
            ax.set_title("No Activity Data")

        # 6. Academic Support Indicators
        ax = axes[2, 1]

        performance = result["performance"]

        indicators = {
        "Low Average": performance["average_grade"] < 60,
        "Failed Grades": performance["failed_records"] > 0,
        "Low Attendance": (
            activity["attendance_pct"] is not None
            and activity["attendance_pct"] < 75
        ),
        "Low Assignments": (
            activity["assignment_completion_pct"] is not None
            and activity["assignment_completion_pct"] < 70
        )
        }

        labels = list(indicators.keys())
        values = [int(x) for x in indicators.values()]

        ax.barh(labels, values)
        ax.set_xticks([0, 1], ["Normal", "Triggered"])
        ax.set_xlim(0, 1.5)
        ax.set_title("Academic Support Indicators")
        ax.grid(axis="x", alpha=0.3)

        # Final layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        plt.close(fig)

    def analyze_student(self, student_id):
        student_rows = self.students[
            self.students["student_id"].astype(str) == str(student_id)
        ]

        if student_rows.empty:
            raise ValueError(f"Student ID '{student_id}' was not found.")

        student = student_rows.iloc[0]
        grades = self._student_grades(student_id)

        if grades.empty:
            raise ValueError(
                f"No valid grade records found for '{student_id}'."
            )

        semester_trends = []

        if "semester" in grades.columns:
            averages = (
                grades.groupby("semester")["numeric_grade"]
                .mean()
                .reset_index()
            )

            order = {
                "Fall 2024": 1,
                "Spring 2025": 2,
                "Fall 2025": 3,
                "Spring 2026": 4,
            }

            averages["order"] = (
                averages["semester"].map(order).fillna(999)
            )
            averages = averages.sort_values("order")

            for _, row in averages.iterrows():
                semester_trends.append({
                    "semester": str(row["semester"]),
                    "average_grade": round(
                        float(row["numeric_grade"]), 2
                    ),
                })

        trend_change = None
        if len(semester_trends) >= 2:
            trend_change = round(
                semester_trends[-1]["average_grade"]
                - semester_trends[-2]["average_grade"],
                2,
            )

        average_grade = float(grades["numeric_grade"].mean())
        passed = int((grades["numeric_grade"] >= 60).sum())
        failed = int((grades["numeric_grade"] < 60).sum())

        course_results = (
            grades.groupby("course_id")["numeric_grade"]
            .mean()
            .reset_index()
            .merge(
                self.courses[["course_id", "course_name"]],
                on="course_id",
                how="left",
            )
            .sort_values("numeric_grade", ascending=False)
        )

        strongest = course_results.iloc[0]
        weakest = course_results.iloc[-1]

        activity = self.activity[
            self.activity["student_id"].astype(str) == str(student_id)
        ].copy()

        activity_summary = {}

        for column in [
            "attendance_pct",
            "study_hours_per_week",
            "quiz_avg",
            "assignment_completion_pct",
        ]:
            if column in activity.columns and not activity.empty:
                values = pd.to_numeric(
                    activity[column], errors="coerce"
                ).dropna()
                activity_summary[column] = (
                    round(float(values.mean()), 2)
                    if not values.empty else None
                )
            else:
                activity_summary[column] = None

        support_reasons = []

        if average_grade < 60:
            support_reasons.append("Average grade is below 60.")
        if failed > 0:
            support_reasons.append(
                f"{failed} historical grade record(s) below 60."
            )

        attendance = activity_summary["attendance_pct"]
        if attendance is not None and attendance < 75:
            support_reasons.append("Average attendance is below 75%.")

        completion = activity_summary["assignment_completion_pct"]
        if completion is not None and completion < 70:
            support_reasons.append(
                "Assignment completion is below 70%."
            )

        return {
            "student": {
                "student_id": str(student["student_id"]),
                "student_name": str(student["student_name"]),
                "department_name": str(student["department_name"]),
                "year_level": int(student["year_level"]),
                "gpa": float(student["gpa"]),
            },
            "performance": {
                "average_grade": round(average_grade, 2),
                "courses_with_grades": int(
                    grades["course_id"].nunique()
                ),
                "passed_records": passed,
                "failed_records": failed,
                "pass_rate_pct": round(
                    passed / len(grades) * 100, 2
                ),
            },
            "strongest_course": {
                "course_name": str(strongest["course_name"]),
                "average_grade": round(
                    float(strongest["numeric_grade"]), 2
                ),
            },
            "weakest_course": {
                "course_name": str(weakest["course_name"]),
                "average_grade": round(
                    float(weakest["numeric_grade"]), 2
                ),
            },
            "semester_trends": semester_trends,
            "trend_change": trend_change,
            "activity": activity_summary,
            "support_reasons": support_reasons,
            "note": (
                "Results use synthetic historical records. Support "
                "indicators are transparent prototype rules, not a "
                "validated prediction of student outcomes."
            ),
        }

    def print_report(self, student_id):
        result = self.analyze_student(student_id)
        student = result["student"]
        performance = result["performance"]

        print(f"\nAcademic Analytics: {student['student_name']}")
        print(
            f"Department: {student['department_name']} | "
            f"Year: {student['year_level']} | "
            f"GPA: {student['gpa']:.2f}/4.00"
        )

        print("\n--- Academic Performance ---")
        print(f"Average grade: {performance['average_grade']:.2f}/100")
        print(f"Courses with grades: {performance['courses_with_grades']}")
        print(f"Passing records: {performance['passed_records']}")
        print(f"Below-passing records: {performance['failed_records']}")
        print(f"Pass rate: {performance['pass_rate_pct']:.2f}%")

        print("\n--- Subject Strengths and Weaknesses ---")
        print(
            f"Strongest: {result['strongest_course']['course_name']} "
            f"({result['strongest_course']['average_grade']:.2f})"
        )
        print(
            f"Needs review: {result['weakest_course']['course_name']} "
            f"({result['weakest_course']['average_grade']:.2f})"
        )

        print("\n--- Semester Performance ---")
        if result["semester_trends"]:
            for item in result["semester_trends"]:
                print(
                    f"{item['semester']}: "
                    f"{item['average_grade']:.2f}/100"
                )
        else:
            print("No semester trend data available.")

        change = result["trend_change"]
        if change is not None:
            if change > 0:
                print(f"Performance improved by {change:.2f} points.")
            elif change < 0:
                print(
                    f"Performance declined by {abs(change):.2f} points."
                )
            else:
                print("Performance stayed the same.")
        else:
            print("Not enough semester data to calculate a trend.")

        print("\n--- Study Activity Averages ---")
        labels = {
            "attendance_pct": "Attendance (%)",
            "study_hours_per_week": "Study hours per week",
            "quiz_avg": "Quiz average",
            "assignment_completion_pct": "Assignment completion (%)",
        }

        for key, label in labels.items():
            value = result["activity"][key]
            print(f"{label}: {value if value is not None else 'No data'}")

        print("\n--- Academic Support Indicators ---")
        if result["support_reasons"]:
            for reason in result["support_reasons"]:
                print(f"- {reason}")
        else:
            print("No support indicators triggered.")

        print(f"\nNote: {result['note']}")
        print("\n--- Displaying plots one at a time ---")
        self.create_dashboard(student_id)   

if __name__ == "__main__":
    student_id = input("Student ID (for example S0003): ").strip()

    try:
        analytics = AcademicAnalytics()
        analytics.print_report(student_id)
    except (ValueError, FileNotFoundError) as error:
        print(f"Could not analyze student: {error}")



