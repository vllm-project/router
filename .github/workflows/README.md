# GitHub Actions CI/CD Setup

This directory contains GitHub Actions workflows that ensure CI/CD runs for **all pull requests**, including those from external contributors and forks.

## Why GitHub Actions + Buildkite?

**Problem**: Buildkite webhooks don't trigger for PRs from forks or external contributors due to security restrictions.

**Solution**: GitHub Actions runs on all PRs and triggers Buildkite builds via API, ensuring consistent CI/CD coverage.

## Workflows

### `ci.yml` - Main CI Workflow

Runs on all PRs and pushes to main branches. Includes:

1. **Buildkite Trigger**: Triggers the main Buildkite pipeline for comprehensive testing
2. **Fast Checks**: Runs quick format/lint checks directly in GitHub Actions for immediate feedback

## Setup Instructions

### 1. Create Buildkite API Token

1. Go to **Buildkite** → **Settings** → **API Access Tokens**
2. Click **New API Access Token**
3. Name it: `GitHub Actions CI Trigger`
4. Scopes required:
   - `read_builds`
   - `write_builds`
5. Copy the generated token

### 2. Find Your Buildkite Organization and Pipeline Slugs

Your Buildkite URLs look like: `https://buildkite.com/{org}/{pipeline}/builds`

- **Organization slug**: The `{org}` part of your Buildkite URL
- **Pipeline slug**: The `{pipeline}` part of your Buildkite URL

Example: If your URL is `https://buildkite.com/my-company/vllm-router/builds`
- Org: `my-company`
- Pipeline: `vllm-router`

### 3. Add GitHub Repository Secrets

1. Go to your GitHub repository → **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret** and add these three secrets:

| Secret Name | Value | Example |
|-------------|-------|---------|
| `BUILDKITE_API_TOKEN` | Your Buildkite API token from step 1 | `bkua_abc123...` |
| `BUILDKITE_ORG` | Your Buildkite organization slug | `my-company` |
| `BUILDKITE_PIPELINE` | Your Buildkite pipeline slug | `vllm-router` |

### 4. Verify Setup

1. Create a test PR or push to main
2. Check the **Actions** tab in GitHub - you should see the workflow running
3. The workflow will trigger a Buildkite build
4. Check your Buildkite dashboard for the triggered build

## Workflow Triggers

The CI workflow runs on:
- **Pull requests**: `opened`, `synchronize` (new commits), `reopened`
- **Push to branches**: `main`, `master`, `develop`
- **Manual trigger**: Via GitHub Actions UI (for testing)

## Fast Checks in GitHub Actions

To provide quick feedback, the workflow runs these checks directly in GitHub Actions:
- Rust format check (`cargo fmt`)
- Clippy linting (`cargo clippy`)
- Python format check (black, ruff)

These run in parallel with the Buildkite trigger, giving developers immediate feedback while comprehensive tests run on Buildkite.

## Troubleshooting

### "BUILDKITE_API_TOKEN not configured" message

The workflow is skipping Buildkite trigger because secrets aren't set up. Follow the setup instructions above.

### Buildkite build not triggering

1. Verify secrets are correct:
   - Check secret names match exactly (case-sensitive)
   - Verify Buildkite org and pipeline slugs are correct
2. Check Buildkite API token permissions
3. View workflow logs in GitHub Actions for error details

### Build triggered but not showing status

Buildkite needs to be configured to report status back to GitHub:
1. Go to Buildkite pipeline settings
2. Enable **GitHub Commit Status** integration
3. Enable **GitHub Pull Request** integration

## Benefits of This Setup

✅ **Universal PR support**: Works for PRs from forks, external contributors, and organization members
✅ **Quick feedback**: Fast checks run in GitHub Actions for immediate results
✅ **Comprehensive testing**: Full test suite runs on Buildkite infrastructure
✅ **No manual intervention**: Fully automated for all contributors
✅ **Secure**: External contributors don't need access to Buildkite secrets

## Alternative: Buildkite GitHub App

If you prefer, you can use the [Buildkite GitHub App](https://github.com/apps/buildkite) instead, which handles webhooks automatically. However, this approach gives you more control and works better with fork PRs.

## Additional Resources

- [Buildkite API Documentation](https://buildkite.com/docs/apis/rest-api)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Buildkite GitHub Integration](https://buildkite.com/docs/integrations/github)
