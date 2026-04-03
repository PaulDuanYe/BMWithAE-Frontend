// 轮询job进度
let pollInterval = null;
let isPolling = false;

function pollJobProgress(protectedAttrs) {
    if (pollInterval) clearInterval(pollInterval);

    if (state.runDemo) protectedAttrs = ['SEX', 'RACE',];
    pollInterval = setInterval(async () => {
        if (isPolling) return;
        isPolling = true;

        try {
            const result = await api.getDemoJobStatus(state.jobId);
            // update only if current_iteration has increased
            if (result.finished_iteration >= state.currentStep) {
                // 後續可以刪，後端直接傳threshold
                if (state.currentStep === 0) {
                    state.epsilonThreshold = 0;

                for (const attr of protectedAttrs) {
                    const attrData = getCaseInsensitive(result.data.iterations[0].epsilon_values, attr);
                    const values = Object.values(attrData?.epsilon_values || {});
                    if (!values.length) continue;

                    const avg = values.reduce((sum, v) => sum + v, 0) / values.length;
                    const epsilon = avg * config.mainThresholdEpsilon;

                    state.epsilonThreshold = Math.max(epsilon, state.epsilonThreshold);
                }
                }

                state.currentStep = result.finished_iteration + 1;
                if (state.runMode === "step"){
                    clearInterval(pollInterval);
                    pollInterval = null;
                    pauseProcess();
                }
                renderProgress(result.data);
            }

            if (result.job_status === "completed") {
                localStorage.setItem("data", JSON.stringify(result.data));
                localStorage.setItem("protectedAttrs", JSON.stringify(protectedAttrs));
                console.log("data:", JSON.stringify(result.data));
                console.log("protectedAttrs:", JSON.stringify(protectedAttrs));
                clearInterval(pollInterval);
                pollInterval = null;
                completeProcess(result.log);
            }
        } catch (err) {
            clearInterval(pollInterval);
            pollInterval = null;
            processError(err);
        } finally {
            isPolling = false;
        }
    }, 1000);
}