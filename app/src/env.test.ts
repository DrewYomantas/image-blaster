import { describe, it, expect } from 'vitest'
import dotenv from 'dotenv'
import path from 'path'
import fs from 'node:fs'

const example = dotenv.parse(fs.readFileSync(path.resolve(__dirname, '../../.env.example')))

describe('environment template', () => {
  it('documents the World Labs key without requiring credentials', () => {
    expect(example.WORLD_LABS_API_KEY).toBe('your_world_labs_key_here')
  })

  it('documents the FAL key without requiring credentials', () => {
    expect(example.FAL_KEY).toBe('your_fal_key_here')
  })
})
