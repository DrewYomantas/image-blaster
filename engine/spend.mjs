import { AsyncLocalStorage } from "node:async_hooks";

const approvals = new AsyncLocalStorage();
let environmentApproval;

export function assertSpendPolicy(policy) {
  if (policy?.allowPaid !== true || !Number.isFinite(policy.maxCostUSD) || policy.maxCostUSD <= 0 ||
      !Number.isFinite(policy.estimatedCostUSD) || policy.estimatedCostUSD <= 0 ||
      policy.estimatedCostUSD > policy.maxCostUSD || typeof policy.approvalReference !== "string" || !policy.approvalReference.trim() ||
      !Array.isArray(policy.endpoints) || !policy.endpoints.length || policy.endpoints.some((endpoint) => typeof endpoint !== "string" || !endpoint)) {
    throw new Error("Paid generation blocked: explicit approval, endpoint, positive cost estimate and sufficient USD budget are required.");
  }
}

export function withSpendPolicy(policy, action) {
  assertSpendPolicy(policy);
  return approvals.run({ ...policy, remainingUSD: policy.maxCostUSD }, action);
}

export function assertPaidSubmission(endpoint) {
  let policy = approvals.getStore();
  if (!policy) {
    if (!environmentApproval) {
      const candidate = {
        allowPaid: process.env.IMAGE_BLASTER_ALLOW_PAID === "1",
        maxCostUSD: Number(process.env.IMAGE_BLASTER_MAX_COST_USD),
        estimatedCostUSD: Number(process.env.IMAGE_BLASTER_ESTIMATED_COST_USD),
        approvalReference: process.env.IMAGE_BLASTER_APPROVAL,
        endpoints: [process.env.IMAGE_BLASTER_PAID_ENDPOINT]
      };
      assertSpendPolicy(candidate);
      environmentApproval = { ...candidate, remainingUSD: candidate.maxCostUSD };
    }
    policy = environmentApproval;
  }
  if (!policy.endpoints.includes(endpoint) || policy.remainingUSD < policy.estimatedCostUSD) throw new Error("Paid generation blocked: endpoint or remaining budget does not match approval.");
  policy.remainingUSD -= policy.estimatedCostUSD;
}
